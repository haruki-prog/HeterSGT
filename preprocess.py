#!/usr/bin/env python3
"""Preprocess raw dataset files into the HeteroSGT graph input format."""

import argparse
import importlib
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ensure_spacy_model(model_name='en_core_web_sm'):
    try:
        spacy_module = importlib.import_module('spacy')
    except Exception:
        return None
    try:
        return spacy_module.load(model_name)
    except OSError:
        try:
            import spacy.cli as spacy_cli
            spacy_cli.download(model_name)
            return spacy_module.load(model_name)
        except Exception:
            return None


def mean_pooling(model_output, attention_mask):
    import torch

    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_texts_with_bert(texts, model_name='bert-base-uncased', max_len=32,
                           batch_size=32, device='auto'):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError('transformers is required for BERT entity/topic features.') from exc

    import torch

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    batches = []
    total = len(texts)
    for start in range(0, total, batch_size):
        encoded = tokenizer(
            list(texts[start:start + batch_size]),
            padding=True, truncation=True, max_length=max_len, return_tensors='pt',
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            batches.append(mean_pooling(model(**encoded), encoded['attention_mask']).cpu().numpy())
        print(f'BERT features: {min(start + batch_size, total)}/{total}', end='\r', flush=True)
    print(f'BERT features: {total}/{total} complete', flush=True)
    return np.concatenate(batches).astype(np.float64)


def align_embedding_dim(embeddings, target_dim):
    if embeddings.shape[1] == target_dim:
        return embeddings
    if embeddings.shape[1] < target_dim:
        pad = np.zeros((embeddings.shape[0], target_dim - embeddings.shape[1]), dtype=np.float64)
        return np.concatenate([embeddings, pad], axis=1)
    return embeddings[:, :target_dim]


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_news_dataframe(raw_news_path):
    df = pd.read_csv(raw_news_path, encoding='utf-8')
    if 'news_id' not in df.columns:
        raise ValueError('Raw dataset must contain a news_id column.')
    if 'reliability' not in df.columns:
        raise ValueError('Raw dataset must contain a reliability column for labels.')

    df = df.copy()
    df['news_id'] = df['news_id'].astype(int)
    df = df.sort_values('news_id').reset_index(drop=True)
    if 'label' not in df.columns:
        df['label'] = df['reliability'].astype(int)
    df['title'] = df['title'].fillna('')
    df['body_text'] = df['body_text'].fillna('')
    df['text'] = (df['title'] + ' ' + df['body_text']).map(clean_text)
    return df


def save_news_final(df, dataset_dir):
    out_path = os.path.join(dataset_dir, 'news_final.xlsx')
    out_df = df.loc[:, ['news_id', 'label', 'url', 'publisher', 'publish_date', 'author', 'title', 'political_bias', 'country', 'reliability']].copy()
    try:
        out_df.to_excel(out_path, index=False, engine='openpyxl')
    except Exception:
        out_df.to_csv(out_path, index=False, encoding='utf-8')
    print(f'Created {out_path}')


def build_news_embeddings(texts, hidden_size, labels, train_indices, epochs=5, batch_size=32, device='auto'):
    """Fit the dual-attention encoder on training labels, then encode every article."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    if hidden_size % 2:
        raise ValueError('hidden_size must be even for bidirectional GRUs.')
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if not len(train_indices) or train_indices.min() < 0 or train_indices.max() >= len(texts):
        raise ValueError('train_indices must identify articles in texts.')
    if len(set(train_indices.tolist())) != len(train_indices):
        raise ValueError('train_indices must not contain duplicates.')
    if len(labels) != len(texts):
        raise ValueError('labels and texts must have the same length.')
    torch.manual_seed(0)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available to PyTorch.')
    print(f'News encoder device: {device}', flush=True)
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    token_pattern = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
    documents = [[token_pattern.findall(sentence.lower())[:64]
                  for sentence in sentence_pattern.split(text)[:32]] for text in texts]
    counts = Counter(word for doc in documents for sentence in doc for word in sentence)
    vocab = {word: i + 2 for i, (word, _) in enumerate(counts.most_common(30000))}
    encoded = []
    for doc in documents:
        sentences = [[vocab.get(word, 1) for word in sentence] for sentence in doc if sentence]
        encoded.append(sentences or [[1]])

    class Attention(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.projection = nn.Linear(size, size)
            self.context = nn.Linear(size, 1, bias=False)

        def forward(self, states, mask):
            scores = self.context(torch.tanh(self.projection(states))).squeeze(-1)
            weights = torch.softmax(scores.masked_fill(~mask, -1e9), dim=-1)
            return (states * weights.unsqueeze(-1)).sum(dim=1)

    class NewsEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(vocab) + 2, hidden_size, padding_idx=0)
            self.word_gru = nn.GRU(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
            self.word_attention = Attention(hidden_size)
            self.sentence_gru = nn.GRU(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
            self.sentence_attention = Attention(hidden_size)
            self.classifier = nn.Linear(hidden_size, len(classes))

        def forward(self, batch):
            batch_size, sentence_count, word_count = batch.shape
            words = batch.reshape(-1, word_count)
            word_states, _ = self.word_gru(self.embedding(words))
            sentences = self.word_attention(word_states, words.ne(0))
            sentences = sentences.reshape(batch_size, sentence_count, -1)
            sentence_states, _ = self.sentence_gru(sentences)
            features = self.sentence_attention(sentence_states, batch.ne(0).any(dim=-1))
            return features, self.classifier(features)

    def collate(indices):
        selected = [encoded[i] for i in indices]
        max_sentences = max(map(len, selected))
        max_words = max(len(sentence) for doc in selected for sentence in doc)
        batch = torch.zeros((len(indices), max_sentences, max_words), dtype=torch.long)
        for row, doc in enumerate(selected):
            for col, sentence in enumerate(doc):
                batch[row, col, :len(sentence)] = torch.tensor(sentence)
        return batch

    classes = {label: i for i, label in enumerate(sorted({labels[i] for i in train_indices}))}
    targets = torch.tensor([classes[labels[i]] for i in train_indices])
    model = NewsEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(train_indices)).split(batch_size):
            article_indices = train_indices[indices.tolist()].tolist()
            _, logits = model(collate(article_indices).to(device))
            loss = F.cross_entropy(logits, targets[indices].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'News encoder epoch {epoch + 1}/{epochs} complete', flush=True)
    model.eval()
    with torch.no_grad():
        features = [model(collate(list(range(start, min(start + batch_size, len(encoded))))).to(device))[0].cpu()
                    for start in range(0, len(encoded), batch_size)]
    return torch.cat(features).numpy().astype(np.float64)


ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
COMMON_ENTITY_BLACKLIST = {
    'The', 'A', 'An', 'In', 'On', 'For', 'And', 'Of', 'At', 'To', 'With', 'By',
    'From', 'Is', 'Are', 'Be', 'Has', 'Will', 'This', 'That', 'It', 'Its', 'As',
    'Not', 'But', 'Or', 'If', 'We', 'Our', 'Your', 'You', 'Their', 'His', 'Her'
}


def extract_entities_from_text(text, nlp=None):
    if nlp is not None:
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in {'PERSON', 'ORG', 'GPE', 'LOC', 'FAC', 'NORP', 'PRODUCT'}:
                token = ent.text.strip()
                if token and not token.isdigit() and token.lower() not in {'the', 'a', 'an'}:
                    entities.append(token)
        return entities

    entities = []
    for match in ENTITY_PATTERN.findall(text):
        token = match.strip()
        if len(token) <= 1 or token in COMMON_ENTITY_BLACKLIST:
            continue
        if token.isdigit():
            continue
        entities.append(token)
    return entities


def build_entity_data(texts):
    nlp = ensure_spacy_model()
    if nlp is None:
        raise RuntimeError('spaCy en_core_web_sm is required for paper-compatible entity extraction.')
    candidate_counts = Counter()
    news_entity_candidates = []
    print(f'Extracting entities: 0/{len(texts)}', flush=True)
    for index, text in enumerate(texts, 1):
        candidates = extract_entities_from_text(text, nlp)
        news_entity_candidates.append(candidates)
        candidate_counts.update(candidates)
        if index % 100 == 0 or index == len(texts):
            print(f'Extracting entities: {index}/{len(texts)}', flush=True)

    selected_entities = sorted(candidate_counts)
    entity_to_id = {ent: idx for idx, ent in enumerate(selected_entities)}
    news2entity = []
    news_entity_mapping = []
    for i, candidates in enumerate(news_entity_candidates):
        connected = sorted({entity_to_id[ent] for ent in candidates})
        news_entity_mapping.append(connected)
        news2entity.extend((i, ent_id) for ent_id in connected)

    return selected_entities, news2entity, news_entity_mapping


def build_entity_embeddings(entity_names, hidden_size, device='auto'):
    if not entity_names:
        return np.zeros((0, hidden_size), dtype=np.float64)

    entity_texts = [name for name in entity_names]
    bert_embeddings = encode_texts_with_bert(entity_texts, device=device)
    return align_embedding_dim(bert_embeddings, hidden_size)


def build_topic_embeddings(topic_descriptions, hidden_size, device='auto'):
    if not topic_descriptions:
        return np.zeros((0, hidden_size), dtype=np.float64)
    topic_embeddings = encode_texts_with_bert(topic_descriptions, device=device)
    return align_embedding_dim(topic_embeddings, hidden_size)


def fit_lda_cuda(doc_term, num_topics, max_iter=100, tolerance=1e-3):
    """Batch variational LDA using CUDA; returns topic-word and document-topic matrices."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA LDA was requested but PyTorch cannot access CUDA.')
    matrix = doc_term.tocoo()
    device = torch.device('cuda')
    docs = torch.as_tensor(matrix.row.astype(np.int64), device=device)
    words = torch.as_tensor(matrix.col.astype(np.int64), device=device)
    counts = torch.as_tensor(matrix.data, device=device, dtype=torch.float32)
    n_docs, n_words = matrix.shape
    alpha = 1.0 / num_topics
    eta = 1.0 / num_topics
    torch.manual_seed(0)
    components = torch.rand((num_topics, n_words), device=device) + eta
    lengths = torch.zeros(n_docs, device=device)
    lengths.index_add_(0, docs, counts)
    gamma = (alpha + lengths[:, None] / num_topics).expand(-1, num_topics).clone()
    word_log_likelihood = None

    with torch.no_grad():
        for iteration in range(max_iter):
            word_log_likelihood = torch.digamma(components) - torch.digamma(
                components.sum(dim=1, keepdim=True)
            )
            # Coordinate updates of document-topic posteriors.
            for _ in range(3):
                doc_log_likelihood = torch.digamma(gamma) - torch.digamma(
                    gamma.sum(dim=1, keepdim=True)
                )
                responsibilities = torch.softmax(
                    doc_log_likelihood[docs] + word_log_likelihood[:, words].T,
                    dim=1,
                )
                gamma_new = torch.full_like(gamma, alpha)
                gamma_new.index_add_(0, docs, counts[:, None] * responsibilities)
                gamma = gamma_new

            updated = torch.full_like(components, eta)
            weighted = counts[:, None] * responsibilities
            updated.T.index_add_(0, words, weighted)
            change = ((updated - components).abs().sum() /
                      components.abs().sum().clamp_min(1)).item()
            components = updated
            if (iteration + 1) % 5 == 0 or iteration == 0 or change < tolerance:
                print(f'LDA CUDA iteration {iteration + 1}/{max_iter}, change={change:.5f}', flush=True)
            if change < tolerance:
                break

    proportions = gamma / gamma.sum(dim=1, keepdim=True)
    return components.cpu().numpy(), proportions.cpu().numpy()


def select_lda_topic_count(texts, min_topics=10, max_topics=50, step=5, device='auto'):
    """Select the topic count by UMass coherence, as in the paper's evaluation."""
    vectorizer = CountVectorizer(max_features=5000, stop_words='english', lowercase=True)
    matrix = vectorizer.fit_transform(texts)
    binary = matrix.astype(bool).astype(int)
    document_frequency = np.asarray(binary.sum(axis=0)).ravel()
    candidates = range(min_topics, min(max_topics, len(texts)) + 1, step)
    scores = {}
    for k in candidates:
        if device == 'cuda' or (device == 'auto' and __import__('torch').cuda.is_available()):
            components, _ = fit_lda_cuda(matrix, k)
        else:
            lda = LatentDirichletAllocation(n_components=k, random_state=0)
            lda.fit(matrix)
            components = lda.components_
        coherence = []
        for topic in components:
            words = np.argsort(topic)[-10:][::-1]
            for i, later in enumerate(words):
                for earlier in words[:i]:
                    both = binary[:, later].multiply(binary[:, earlier]).sum()
                    coherence.append(np.log((both + 1) / max(document_frequency[earlier], 1)))
        scores[k] = float(np.mean(coherence))
    if not scores:
        raise ValueError('Not enough articles to estimate a topic count; pass --num_topics.')
    return max(scores, key=scores.get)


def build_topic_data(embeddings, num_topics, texts=None, topn=3, device='auto'):
    """LDA-based topic extraction aligned with the paper design.

    Each news item is assigned to the top-k topics according to its LDA topic
    distribution. The same topic relation is then used when building news-news
    links and news-topic edges. Topic node features are initialized from BERT
    embeddings of their top words.
    """
    if texts is None:
        raise ValueError('texts are required for LDA topic extraction.')

    vectorizer = CountVectorizer(max_features=5000, stop_words='english', lowercase=True)
    doc_term = vectorizer.fit_transform(texts)
    if device == 'cuda' or (device == 'auto' and __import__('torch').cuda.is_available()):
        components, news_topic_distribution = fit_lda_cuda(doc_term, num_topics)
        lda = components
    else:
        lda = LatentDirichletAllocation(
            n_components=num_topics, max_iter=1000, learning_method='batch',
            random_state=0, verbose=1,
        )
        lda.fit(doc_term)
        components = lda.components_
        news_topic_distribution = lda.transform(doc_term)

    topic_word_index = np.asarray(vectorizer.get_feature_names_out())
    topic_descriptions = []
    for topic_idx in range(num_topics):
        top_word_idx = np.argsort(components[topic_idx])[::-1][:10]
        top_words = [topic_word_index[i] for i in top_word_idx]
        topic_descriptions.append(' '.join(top_words))

    news2topic = []
    for i in range(len(texts)):
        top_topics = np.argsort(news_topic_distribution[i])[::-1][:topn]
        for topic_id in top_topics:
            news2topic.append((i, int(topic_id)))

    return lda, vectorizer, topic_descriptions, news2topic


def build_news2news_edges_by_metadata(news_entity_mapping, news_topic_mapping):
    """Build news-news links from metadata relations.

    According to the paper definition, two news items are linked when:
    1) they share more entities than the average shared-entity count across all
       news pairs, or
    2) they are assigned to the same topic.
    """
    n_news = len(news_entity_mapping)
    if n_news < 2:
        return []

    entity_sets = [set(entity_ids) for entity_ids in news_entity_mapping]
    topic_sets = [set(topic_ids) for topic_ids in news_topic_mapping]

    shared_counts = []
    for i in range(n_news):
        for j in range(i + 1, n_news):
            shared_counts.append(len(entity_sets[i].intersection(entity_sets[j])))

    avg_shared = float(np.mean(shared_counts)) if shared_counts else 0.0
    edges = set()

    for i in range(n_news):
        for j in range(i + 1, n_news):
            shared_count = len(entity_sets[i].intersection(entity_sets[j]))
            same_topic = bool(topic_sets[i].intersection(topic_sets[j]))
            if shared_count > avg_shared or same_topic:
                edges.add((i, j))

    return sorted(edges)


def make_index_dict(prefix, count):
    mapping = {}
    for idx in range(count):
        name = f'{prefix}_{idx}'
        mapping[name] = idx
        mapping[idx] = name
    return mapping


def build_global_index(news_count, entity_count, topic_count):
    global_index = {}
    next_id = 0
    for idx in range(news_count):
        global_index[f'news_{idx}'] = next_id
        next_id += 1
    for idx in range(entity_count):
        global_index[f'entity_{idx}'] = next_id
        next_id += 1
    for idx in range(topic_count):
        global_index[f'topic_{idx}'] = next_id
        next_id += 1
    return global_index


def save_numpy(path, array):
    np.save(path, array)
    print(f'  saved {path} ({array.shape})')


def save_mapping(path, mapping):
    np.save(path, mapping, allow_pickle=True)
    print(f'  saved {path} ({len(mapping)} entries)')


def save_edge_csv(path, edges, columns):
    df = pd.DataFrame(edges, columns=columns)
    df.to_csv(path, index=False, encoding='utf-8')
    print(f'  saved {path} ({len(edges)} edges)')


def save_edge_excel(path, edges, columns):
    df = pd.DataFrame(edges, columns=columns)
    df.to_excel(path, index=False, engine='openpyxl')
    print(f'  saved {path} ({len(edges)} edges)')


def main():
    parser = argparse.ArgumentParser(description='Preprocess raw dataset files for HeteroSGT.')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--hiddenSize', type=int, default=600)
    parser.add_argument('--num_topics', type=int, default=None)
    parser.add_argument('--topn', type=int, default=3)
    parser.add_argument('--news_epochs', type=int, default=5)
    parser.add_argument('--news_batch_size', type=int, default=32)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--skip_entity_embeddings', action='store_true',
                        help='Reuse existing entity embeddings after validating their shape.')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(repo_root, '..', 'Data', args.dataset)
    raw_news_path = os.path.join(dataset_dir, 'recovery-news-data.csv')
    if not os.path.exists(raw_news_path):
        raise FileNotFoundError(f'Could not find raw news file: {raw_news_path}')
    ensure_dir(dataset_dir)
    graph_nodes_dir = os.path.join(dataset_dir, 'graph', 'nodes')
    graph_edges_dir = os.path.join(dataset_dir, 'graph', 'edges')
    ensure_dir(graph_nodes_dir)
    ensure_dir(graph_edges_dir)

    news_df = build_news_dataframe(raw_news_path)
    save_news_final(news_df, dataset_dir)

    text_inputs = news_df['text'].tolist()
    # Match Data.FakenewsDataset.get_train_val_test_id() exactly. Validation
    # and test articles are encoded, but neither set's labels train the encoder.
    train_val_indices, test_indices = train_test_split(
        np.arange(len(news_df)), test_size=0.1, random_state=0,
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.1 / 0.9, random_state=0,
    )
    print(f'News encoder labels: {len(train_indices)} train; '
          f'{len(val_indices)} validation and {len(test_indices)} test labels excluded',
          flush=True)
    news_embeddings = build_news_embeddings(
        text_inputs, args.hiddenSize, news_df['label'].tolist(), train_indices,
        epochs=args.news_epochs, batch_size=args.news_batch_size, device=args.device,
    )
    save_numpy(
        os.path.join(graph_nodes_dir, f'news_embeddings_{args.hiddenSize}_final.npy'),
        news_embeddings,
    )

    entity_names, news2entity_pairs, news_entity_mapping = build_entity_data(text_inputs)
    num_entities = len(entity_names)
    entity_embedding_path = os.path.join(graph_nodes_dir, f'entity_embeddings_{args.hiddenSize}.npy')
    if args.skip_entity_embeddings:
        entity_embeddings = np.load(entity_embedding_path)
        expected_shape = (num_entities, args.hiddenSize)
        if entity_embeddings.shape != expected_shape:
            raise ValueError(f'Existing entity embeddings have shape {entity_embeddings.shape}; expected {expected_shape}.')
        print(f'Reusing {entity_embedding_path} ({entity_embeddings.shape})', flush=True)
    else:
        print(f'Encoding {num_entities} entities with BERT', flush=True)
        entity_embeddings = build_entity_embeddings(entity_names, args.hiddenSize, device=args.device)
        save_numpy(entity_embedding_path, entity_embeddings)

    paper_topic_counts = {'MM COVID': 50, 'ReCOVery': 35, 'MC Fake': 40, 'LIAR': 50, 'PAN2020': 25}
    num_topics = args.num_topics or paper_topic_counts.get(args.dataset)
    if num_topics is None:
        num_topics = select_lda_topic_count(text_inputs, device=args.device)
    print(f'Fitting LDA with {num_topics} topics', flush=True)
    _, _, topic_descriptions, news2topic_pairs = build_topic_data(
        news_embeddings,
        num_topics,
        texts=text_inputs,
        topn=args.topn,
        device=args.device,
    )
    print('Encoding topics with BERT', flush=True)
    topic_embeddings = build_topic_embeddings(topic_descriptions, args.hiddenSize, device=args.device)
    save_numpy(os.path.join(graph_nodes_dir, f'topic_embeddings_{args.hiddenSize}.npy'), topic_embeddings)

    news_topic_mapping = [set() for _ in range(len(news_df))]
    for news_idx, topic_idx in news2topic_pairs:
        news_topic_mapping[news_idx].add(topic_idx)

    print('Building news-news edges', flush=True)
    news2news_pairs = build_news2news_edges_by_metadata(
        news_entity_mapping,
        news_topic_mapping,
    )
    save_edge_csv(os.path.join(graph_edges_dir, 'news2news.csv'), [(f'news_{a}', f'news_{b}') for a, b in news2news_pairs], ['news_id', 'news_id_2'])
    save_edge_excel(os.path.join(graph_edges_dir, 'news2entity.xlsx'), [(f'news_{n}', f'entity_{e}') for n, e in news2entity_pairs], ['news_id', 'entity_id'])
    save_edge_excel(os.path.join(graph_edges_dir, 'news2topic.xlsx'), [(f'news_{n}', f'topic_{t}') for n, t in news2topic_pairs], ['news_id', 'topic_id'])

    news_index = make_index_dict('news', len(news_df))
    save_mapping(os.path.join(graph_nodes_dir, 'news_index.npy'), news_index)
    entity_index = make_index_dict('entity', num_entities)
    save_mapping(os.path.join(graph_nodes_dir, 'entity_index.npy'), entity_index)
    topic_index = make_index_dict('topic', num_topics)
    save_mapping(os.path.join(graph_nodes_dir, 'topic_index.npy'), topic_index)

    global_index = build_global_index(len(news_df), num_entities, num_topics)
    save_mapping(os.path.join(graph_nodes_dir, 'global_index_graph1.npy'), global_index)

    print('Preprocessing finished.')


if __name__ == '__main__':
    main()
