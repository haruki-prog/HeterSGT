# HeterSGT システムアーキテクチャ ガイド

このドキュメントは、HeterSGTフォルダ内の各ソースファイルの役割と相互関係を体系的にまとめたものです。

---

## 📁 ファイル構成概要

```
HeterSGT/
├── main.py                      # エントリーポイント
├── model/
│   ├── model1.py                # モデル定義と訓練ループ
│   └── Base.py                  # Transformer基盤実装
├── Data.py                      # データセット定義
├── news_RandomWalk.py           # ランダムウォーク生成
├── build_graph.py               # グラフ構築・前処理
├── preprocess.py                # テキスト埋め込み生成
├── utils.py                     # ユーティリティ関数
└── config/
    └── HeteroSGT.json           # ハイパーパラメータ設定
```

---

## 🔄 処理フロー (全体構造)

```
セットアップフェーズ
  ↓
  ├─ build_graph.py
  │   ├→ グラフノード (news, entity, topic) 作成
  │   ├→ グラフエッジ (関係) 定義
  │   └→ .pt ファイルに保存
  │
  └─ preprocess.py
      ├→ テキストを埋め込み化（理論：Dual-attention BiGRU、実装：BERT 事前固定抽出）
      └→ news_embeddings_*.npy に保存（特徴量初期化用）
  
　　　　↓

訓練フェーズ
  ↓
  └─ main.py (エントリーポイント)
      ├→ config/HeteroSGT.json を読み込み
      ├→ utils.load_data()
      │   ├→ グラフ (.pt) を読み込み
      │   ├→ news_RandomWalk.rand_walk()
      │   │   └→ ウォーク・ラベル・タイプ列生成
      │   └→ Data.FakenewsDataset
      │       └→ train/test分割
      ├→ model/model1.Model1() インスタンス化
      │   └→ model/Base.TransformerEncoder 内部初期化
      └→ model/model1.train()
          ├→ 各エポックで：
          │   ├→ model/Base.genX()
          │   │   └→ ウォーク→テンソル変換
          │   ├→ model/model1.Model1.forward()
          │   │   └→ model/Base.TransformerEncoder.forward()
          │   │       ├→ PositionalEncoding
          │   │       ├→ EncoderBlock (複数層)
          │   │       │   ├→ MultiHeadAttention
          │   │       │   └→ FFN
          │   │       └→ 学習済みノード列返却
          │   ├→ model/Base.FNclf()
          │   │   └→ 分類スコア計算
          │   └→ utils.test_once()
          │       └→ 評価メトリクス計算
          └→ models/model1.train()
              └→ results/ に結果保存
```

---

## 📄 各ファイルの詳細説明

### 1️⃣ **main.py** - 実行エントリーポイント

**概要**：  
プロジェクト全体の実行フローを制御する中枢ファイル。

**主な処理**：
```python
1. コマンドライン引数をパース
2. config/HeteroSGT.json を読み込み
3. utils.load_data() でグラフとデータセットをロード
4. モデルを初期化 ( model/model1.Model1 )
5. optimizer を構築
6. model/model1.train() で訓練開始
```

**出力**：
- 訓練済みモデル
- `results/{model_name}/auc/` に AUC ファイルを保存

**呼び出し元**：
```bash
python main.py --dataset <dataset_name> --model Model1 --walk_length 5
```

---

### 2️⃣ **model/model1.py** - モデル定義と訓練ループ

**概要**：  
ニューラルネットワークモデルの定義と、訓練・評価ループの実装。

**主な構成**：

#### **クラス: Model1** 
```python
def __init__(self, config):
    # TransformerEncoder を初期化
    self.Transformer = TransformerEncoder(...)
    # 分類器を初期化
    self.clf = FNclf(...)

def forward(self, X, device, args):
    # X: サブグラフのウォーク (バッチ×ウォーク長×特徴次元)
    h = self.Transformer(X, self.walk_length, args)
    # → h: Transformer処理済みのノード層 (学習済み表現)
    scores, h_emb = self.clf(h)
    # → scores: 分類スコア, h_emb: 埋め込み
    return scores, preds, h_emb[:,0,:]
```

#### **関数: train()**
```python
def train(model, train_data, train_label, test_data, test_label, epochs, optimizer, device, args):
    # エポックループ
    for epoch in range(epochs):
        # 訓練フェーズ
        train_X = genX(train_data, device)
        train_scores, train_preds, _, _ = model(train_X, device, args)
        loss = CrossEntropyLoss(train_scores, train_label)
        loss.backward()
        optimizer.step()
        
        # 評価フェーズ
        test_X = genX(test_data, device)
        test_scores, test_preds, _, _ = model(test_X, device, args)
        test_res, auc = test_once(test_preds, test_scores, test_label)
        
        # 結果保存
        torch.save(auc, f"./results/{args.model}/auc/{args.dataset}_r{args.round}_auc.pt")
```

**役割**：
- 各エポックでの順伝播・逆伝播
- メトリクスの計算と保存
- AUC ファイルの周期的な保存

**依存関係**：
- `from model.Base import TransformerEncoder, FNclf, genX`

---

### 3️⃣ **model/Base.py** - Transformer実装

**概要**：  
HeterSGTの核となるTransformerアーキテクチャと各層の実装。

**主な構成**：

#### **クラス: TransformerEncoder**
```python
def __init__(self, query_size, key_size, value_size, num_hiddens, 
             norm_shape, ffn_num_input, ffn_num_hiddens, 
             num_heads, num_layers, walk_length, dropout):
    self.pos_encoding = PositionalEncoding(...)
    self.blks = nn.Sequential()  # num_layers個のEncoderBlockを積層
    for i in range(num_layers):
        self.blks.add_module("block" + str(i), EncoderBlock(...))

def forward(self, X, valid_lens, *args):
    # ポジション情報付加
    X = self.pos_encoding(X * math.sqrt(self.num_hiddens))
    
    # 複数層のアテンション処理
    for i, blk in enumerate(self.blks):
        X = blk(X, valid_lens=None)
        self.attention_weights[i] = blk.attention.attention.attention_weights
    
    return X  # 学習済みノード列を返す
```

**処理内容**：
1. ウォークシーケンス X を入力
2. ポジション情報を付加
3. 各 EncoderBlock で多層アテンション処理
4. 各層で学習済みのノード表現を返す

#### **クラス: EncoderBlock**
```python
def forward(self, X, valid_lens):
    # MultiHeadAttention を適用
    Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
    # Feed-Forward Network を適用
    return self.addnorm2(Y, self.ffn(Y))
```

#### **クラス: MultiHeadAttention**
```python
def forward(self, queries, keys, values, valid_lens):
    # queries, keys, values を複数ヘッドに分割
    # 各ヘッドで DotProductAttention を計算
    # 結果を統合
    return output
```

#### **クラス: DotProductAttention**
```python
def forward(self, queries, keys, values, valid_lens):
    # スケール付きドット積アテンション
    scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
    attention_weights = masked_softmax(scores, valid_lens)
    return torch.bmm(attention_weights, values)
```

#### **クラス: PositionalEncoding**
```python
def forward(self, X):
    # sin/cos 関数でポジション情報を付加
    return X + self.P[:, :X.shape[1], :]
```

#### **クラス: FNclf**
```python
def forward(self, h):
    # h: (バッチ×ウォーク長×隠れ層次元)
    h = self.fcn1(h).relu()  # 非線形変換
    h_emb = h                 # 埋め込み保存
    return self.fcn2(h), h_emb  # (スコア, 埋め込み)
```

#### **関数: genX(data, device)**
```python
def genX(data, device):
    # ウォークごとに、内部ノード表現を抽出してスタック
    X = []
    for i, walk in enumerate(data.walk_list):
        x = []
        for j, node_id in enumerate(walk):
            node_type = data.type_list[i][j]
            if node_type == 0:  # news
                x.append(data.graph["news"].x[data.inner_list[i][j]])
            elif node_type == 1:  # entity
                x.append(data.graph["entity"].x[data.inner_list[i][j]])
            elif node_type == 2:  # topic
                x.append(data.graph["topic"].x[data.inner_list[i][j]])
        X.append(torch.stack(x))
    return torch.stack(X)
```

**役割**：
- **サブグラフのシーケンスをTransformerで処理**
- ノード列を学習済み表現に変換
- 各層のアテンション重みを記録

**依存関係**：
- PyTorch基盤モジュール (nn.Module, Dropout, LayerNorm など)

---

### 4️⃣ **Data.py** - データセット定義

**概要**：  
訓練・テストデータの管理とデータセット構造を定義。

**主な構成**：

#### **クラス: FakenewsDataset**
```python
def __init__(self, walk_list, inner_list, type_list, label_list, graph):
    self.walk_list = walk_list        # ウォーク のリスト
    self.inner_list = inner_list      # 内部ノードID のリスト
    self.type_list = type_list        # ノードタイプ (0=news, 1=entity, 2=topic)
    self.labels = label_list          # ラベル (真実/偽物)
    self.graph = graph                # グラフ全体

def get_train_id(self, test_ratio):
    # テスト分割: ニュース単位で分割
    train_news, test_news, ... = train_test_split(...)
    
    # ウォークをニュースIDでフィルタ
    train_dataset = FakenewsDataset(...)
    test_dataset = FakenewsDataset(...)
    return train_dataset, test_dataset
```

#### **クラス: Vocab**
テキスト語彙管理 (未使用かもしれません)

#### **クラス: Data**
グラフデータの管理

**役割**：
- ウォーク、タイプ列、ラベルの管理
- 訓練/テスト分割の実施
- PyTorch Dataset インターフェースの提供

**呼び出し元**：
- `utils.load_data()` 内で使用

---

### 5️⃣ **news_RandomWalk.py** - ランダムウォーク生成

**概要**：  
異種グラフ上でランダムウォークを生成し、メタパス列を作成。

**主な処理**：

```python
def rand_walk(dataset, restart_prob, num_laps, walk_length):
    # ニュースノードから開始するランダムウォークを生成
    # 各ウォークについて：
    #   - walk_list: グローバルノードID列
    #   - inner_list: 各ノードタイプ内での内部ID
    #   - type_list: ノードタイプ (0=news, 1=entity, 2=topic)
    #   - labels: ニュースの真偽ラベル
    return walk_list, labels, inner_list, type_list
```

**処理内容**：
1. グラフの隣接行列を読み込み
2. 各newsノードから restart_prob で復帰しながらランダムウォーク
3. walk_length 長のシーケンスを生成
4. グローバルID → 内部ID のマッピングを整理

**役割**：
- **サブグラフのシーケンスを提供**
- メタパス学習の入力を作成
- 異種ノード列を type_list で明示

**呼び出し元**：
- `utils.load_data()` 内で使用

---

### 6️⃣ **build_graph.py** - グラフ構築・前処理

**概要**：  
生のデータセットからグラフを構築し、PyTorch Geometric 形式で保存。

**主な処理**：

```python
def load_edge(dataset, node):
    # news←→entity, news←→topic などの エッジを読み込み

def build_graph(dataset, hiddenSize):
    # news_attr = np.load("news_embeddings_{hiddenSize}_final.npy")
    # entity_attr = np.load(...)
    # topic_attr = np.load(...)
    
    # PyTorch Geometric Data オブジェクトを構築
    data = HeteroData()
    data["news"].x = torch.tensor(news_attr)
    data["entity"].x = torch.tensor(entity_attr)
    data["topic"].x = torch.tensor(topic_attr)
    data["news"].y = torch.tensor(news_labels)
    
    # エッジを追加
    data["news", "connects", "entity"].edge_index = ...
    
    # .pt ファイルに保存
    torch.save(data, f"../Data/{dataset}/graph/{dataset}_{hiddenSize}_final.pt")
```

**役割**：
- グラフノード・エッジの定義
- ノード特徴の読み込み
- PyTorch Geometric 形式で保存

**実行タイミング**：
```bash
python build_graph.py --dataset <dataset_name>
```

---

### 7️⃣ **preprocess.py** - ニューステキスト埋め込み初期化

**概要**：  
パイプライン簡略化とリソース節約のため、事前学習済み BERT を用いてテキスト特徴量を固定抽出する実装上の処理。

**【重要】理論と実装の関係**：

本ドキュメントの後続セクション「3.3 ニューステキスト埋め込みモジュール」で詳述する通り、論文 HeterSGT の完全体理論モデルでは、ニュース表現は **Dual-attention BiGRU** により**エンドツーエンド**で学習される設計になっています。

一方、このリポジトリの `preprocess.py` では以下の簡略化を実施しています：
- **実装上の位置づけ**: 事前学習済み BERT から固定埋め込みを抽出（`news_embeddings_*.npy`）
- **用途**: モデル訓練時の特徴量初期化およびパイプライン計算コスト削減
- **制限**: 固定抽出のため、テキスト特徴の部分はモデル訓練中に更新されない

**主な処理**：

```python
def encode_texts_with_bert(texts, model_name='bert-base-uncased'):
    # テキストを事前学習済み BERT でトークン化・エンコード
    embeddings = model(texts)  # (num_texts, embedding_dim)
    mean_embeddings = mean_pooling(embeddings, attention_mask)
    # 固定埋め込みとして保存
    return mean_embeddings

def mean_pooling(model_output, attention_mask):
    # attention_mask でマスクされた位置を除外して平均化
    # 結果: (num_texts, hidden_size)
```

**役割**：
- テキスト → 固定次元ベクトル変換
- ノード（ニュース、エンティティ、トピック）特徴の初期化
- numpy 形式で保存

**実行タイミング**：
前処理として事前実行 (通常、`build_graph.py` 実行前)

> ⚠️ **【実装上の注意】**  
> 完全な理論モデル再現（Dual-attention BiGRU の完全学習）には、テキスト埋め込み部も含めた全層バックプロパゲーション実装が必要です。現在の実装は計算効率優先での近似処理です。

---

### 7️⃣.5️⃣ **ニューステキスト埋め込みモジュール（Dual-attention BiGRU）** - 論文理論（3.3節）

**【論文設計】本ドキュメント補足**

次の内容は、論文 HeterSGT.pdf 3.3節「Dual-attention News Embedding Module」に基づいています。

#### **概要**

本手法（HeterSGT）では、ニュース記事の言語特徴を捉えるために **Dual-attention BiGRU モジュール**を採用しています。単語レベルおよび文章レベルの2段階で文脈情報を集約し、600次元のニュース表現ベクトル $h_{n_i}$ を生成します。

#### **1. 単語レベル・アテンション（Word-level Attention）**

$$
\alpha_{w_{p,q}} = \frac{\exp(u_w^T \tanh(W_w h_{w_{p,q}} + b_w))}{\sum_q \exp(u_w^T \tanh(W_w h_{w_{p,q}} + b_w))}
$$

$$
h_{s_p} = \sum_q \alpha_{w_{p,q}} \cdot h_{w_{p,q}}
$$

**処理内容**：
- 各文章 $p$ 内の単語シーケンス $\{w_{p,1}, w_{p,2}, \ldots, w_{p,Q}\}$ に対し BiGRU を適用
- 直後のコンテキスト情報 $h_{w_{p,q}}$ を抽出
- 学習可能な単語文脈ベクトル $u_w$ と重みパラメータ（$W_w, b_w$）を用いて単語重要度 $\alpha_{w_{p,q}}$ を算出
- アテンション重み付き平均により、文章ベクトル $h_{s_p}$ を生成

#### **2. 文章レベル・アテンション（Sentence-level Attention）**

$$
\alpha_{s_p} = \frac{\exp(u_s^T \tanh(W_s h_{s_p} + b_s))}{\sum_p \exp(u_s^T \tanh(W_s h_{s_p} + b_s))}
$$

$$
h_{n_i} = \sum_p \alpha_{s_p} \cdot h_{s_p}
$$

**処理内容**：
- 単語レベルで得られた文章ベクト列 $\{h_{s_1}, h_{s_2}, \ldots, h_{s_P}\}$ に対し、さらに別の BiGRU を適用
- 文章間の並び順・文脈を符号化
- 学習可能な文章文脈ベクトル $u_s$ と重みパラメータ（$W_s, b_s$）を用いて文章重要度 $\alpha_{s_p}$ を算出
- アテンション重み付き平均により、ニュース全体の代表ベクトル $h_{n_i}$ を生成（600次元）

#### **3. エンドツーエンドの学習最適化**

これらパラメータ（BiGRU の重み、アテンション行列、文脈ベクトル $u_w, u_s$）は、固定値ではなく、最終的な分類損失（Cross-Entropy Loss、式15）の逆伝播を通じて、全結合層や Transformer とともに最適化されます。

$$
\mathcal{L} = -\sum_{i=1}^{|\mathcal{V}|} y_i \log(\hat{y}_i)
$$

#### **4. 計算効率の観点（Table 8, A.4節）**

- **一般的な BERT/Sentence-BERT 直接利用との比較**:
  - BERT 内の標準 Transformer アテンション: $O(n^2)$ （シーケンス長 $n$ に対して2乗）
  - BiGRU + Dual-attention: $O(n)$ （シーケンス長に対して線形）

- **採用理由**: 
  - ニュース記事は「非常に長いテキスト」（数百～数千単語）を含むため、$O(n^2)$ のアテンション計算は計算量が爆発的に増加
  - BiGRU の線形計算量と、階層的アテンション（単語→文章→ニュース）の組み合わせにより、計算効率と分類精度を両立

#### **【実装注記】**

現在のコードリポジトリ（`preprocess.py`）では、計算リソース節約のため、BERT から抽出した事前固定特徴量（`news_embeddings_*.npy`）を使用しており、**完全な Dual-attention BiGRU の学習ループは実装されていません**。

これは実装上の簡略化であり、論文の完全体アーキテクチャ再現には、テキスト埋め込み部の全層エンドツーエンド学習（BiGRU + Dual-attention パラメータの逆伝播更新）が必要です。

---

### 8️⃣ **utils.py** - ユーティリティ関数

**概要**：  
データロード、評価メトリクス計算などの補助関数を提供。

**主な関数**：

#### **load_data(args)**
```python
def load_data(args):
    # グラフを読み込み
    graph = torch.load(f"../Data/{args.dataset}/graph/{args.dataset}_{args.hiddenSize}_final.pt")
    
    # ランダムウォークを生成
    walk_list, labels, inner_list, type_list = rw.rand_walk(...)
    
    # FakenewsDataset を作成
    dataset = FakenewsDataset(walk_list, inner_list, type_list, labels, graph)
    
    # train/test 分割
    train_data, test_data = dataset.get_train_id(test_ratio=0.1)
    
    return graph, train_data, test_data
```

#### **test_once(preds, scores, y)**
```python
def test_once(preds, scores, y):
    # 精度、precision, recall, F1, AUC を計算
    acc = accuracy_score(y, preds)
    pre = precision_score(y, preds, average="binary")
    recall = recall_score(y, preds, average="binary")
    f1 = f1_score(y, preds, average="binary")
    auc = roc_auc_score(y, scores[:,1])
    return {
        "test_acc": acc,
        "test_pre": pre,
        "test_recall": recall,
        "test_f1": f1,
        "test_auc": auc
    }, auc
```

**役割**：
- データセット読み込み
- 評価メトリクス計算
- 結果の集約

**呼び出し元**：
- `main.py`
- `model/model1.py:train()`

---

### 9️⃣ **config/HeteroSGT.json** - ハイパーパラメータ設定

**概要**：  
データセットごとのモデル設定を定義。

**典型的な構成**：
```json
[
  {
    "dataset": "MM COVID",
    "num_hiddens": 768,
    "num_heads": 12,
    "num_layers": 5,
    "ffn_num_input": 768,
    "ffn_num_hiddens": 2048,
    "key_size": 768,
    "query_size": 768,
    "value_size": 768,
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "norm_shape": [5, 768]
  },
  ...
]
```

**役割**：
- データセット別のハイパーパラメータを集中管理
- JSON から Python Config オブジェクトに変換

**読み込み元**：
- `main.py`

---

## 🔗 ファイル間の依存関係

```
main.py
  ├─→ config/HeteroSGT.json (パラメータ)
  ├─→ utils.py
  │    ├─→ news_RandomWalk.py
  │    └─→ Data.py (← build_graph.py で前処理済み)
  └─→ model/model1.py
       ├─→ model/Base.py
       │    ├─→ TransformerEncoder
       │    ├─→ MultiHeadAttention
       │    ├─→ EncoderBlock
       │    ├─→ FNclf
       │    └─→ genX()
       └─→ utils.py (test_once)

【初期準備フェーズ】
preprocess.py → Generate news_embeddings_*.npy
build_graph.py → Generate *.pt グラフファイル
news_RandomWalk.py → Generate ランダムウォーク
```

---

## 🎯 処理フェイズ別の役割

### **Phase 1: グラフ構築 (One-time)**
- **ファイル**: `preprocess.py`, `build_graph.py`
- **入力**: 生のテキスト、グラフエッジデータ
- **出力**: グラフ (.pt), 埋め込み (npy)

### **Phase 2: ランダムウォーク生成 (Each run)**
- **ファイル**: `news_RandomWalk.py`, `Data.py`
- **入力**: グラフ
- **出力**: ウォーク列、タイプ列、ラベル

### **Phase 3: モデル初期化**
- **ファイル**: `main.py`, `model/model1.py`, `model/Base.py`
- **処理**: モデルの構築と optimizer の初期化

### **Phase 4: 訓練ループ (Per epoch)**
- **ファイル**: `model/model1.py:train()`, `model/Base.py`
- **処理**:
  1. `genX()` でウォーク → テンソル変換
  2. `TransformerEncoder` で処理
  3. `FNclf` で分類
  4. `test_once()` で評価
  5. 結果をファイルに保存

---

## 🚀 実行フロー (ステップバイステップ)

### **ステップ 1: グラフ構築 (初回のみ)**
```bash
python build_graph.py --dataset "MM COVID"
```
→ `../Data/MM COVID/graph/MM COVID_600_final.pt` 生成

### **ステップ 2: モデル訓練**
```bash
python main.py --dataset "MM COVID" --model Model1 --walk_length 5 --hiddenSize 600
```

**内部処理フロー**:
1. `main()` 実行開始
2. `config/HeteroSGT.json` から設定読み込み
3. `utils.load_data()` で:
   - グラフ読み込み
   - `news_RandomWalk.rand_walk()` でウォーク生成
   - `Data.FakenewsDataset` で train/test 分割
4. `model/model1.Model1()` でモデル初期化
5. `model/model1.train()` でエポック訓練:
   - 各エポック:
     - `genX()` でウォーク変換
     - `Model1.forward()` で順伝播
     - `TransformerEncoder.forward()` で Transformer 処理
     - `FNclf.forward()` で分類
     - `criterion()` で損失計算
     - `loss.backward()` で逆伝播
     - `optimizer.step()` でパラメータ更新
     - `test_once()` で評価
     - 最良モデルの AUC を保存

---

## 📊 データフロー (詳細)

```
【入力】
ニューステキスト
  ↓
preprocess.py
  ├→ BERT トークン化・エンコード
  └→ news_embeddings_600_final.npy (num_news × 768)

グラフエッジデータ
  ↓
build_graph.py
  ├→ node 特徴を読み込み
  ├→ edge を定義
  └→ *.pt ファイルに保存

【訓練フェーズ】
*.pt グラフ
  ↓
news_RandomWalk.py
  ├→ walk_list (バッチ × walk_length)
  ├→ inner_list (各ウォーク内の内部ノードID)
  ├→ type_list (各位置のノードタイプ: 0/1/2)
  └→ labels (ニュースの真偽ラベル)
  ↓
Data.FakenewsDataset
  ├→ train_data, test_data に分割
  ↓
model/Base.genX()
  ├→ ウォーク内のノード特徴を抽出
  └→ X (バッチ × walk_length × feature_dim)
  ↓
model/Base.TransformerEncoder.forward()
  ├→ PositionalEncoding (位置情報付加)
  ├→ EncoderBlock ×N層 (MultiHeadAttention + FFN)
  └→ h (バッチ × walk_length × feature_dim) [学習済み]
  ↓
model/Base.FNclf.forward()
  ├→ 隠れ層処理
  └→ scores (バッチ × 2) [クラス確率]
  ↓
CrossEntropyLoss
  └→ 逆伝播で全層のパラメータ更新

【評価フェーズ】
test_scores, test_preds
  ↓
utils.test_once()
  ├→ Accuracy, Precision, Recall, F1, AUC 計算
  └→ results/ に保存
```

---

## 🔑 キーポイント

### 1. **Transformer 処理の全容**
- **入力**: ウォークシーケンス (バッチ×walk_length×特徴)
- **処理**: 位置エンコーディング + 多層マルチヘッドアテンション
- **出力**: 学習済みノード表現 (バッチ×walk_length×特徴)

### 2. **異種ノード処理**
- `type_list` でノードタイプを明示
- `inner_list` で各タイプの内部ID を管理
- グラフから対応する特徴ベクトルを動的に抽出

### 3. **分類戦略**
- `args.case3` で集約方法を選択:
  - `"h0"`: ウォーク最初のノード (newsノード) のみ
  - `"mean"`: ウォーク全体の平均
  - `"max"`: ウォーク全体の最大プーリング

### 4. **訓練の流れ**
- 各エポックで train/test 双方を評価
- 最良 F1 スコア時の AUC をファイルに保存
- 早期終了やモデルチェックポイント機能なし

---

## 📝 まとめテーブル

| ファイル | 主要クラス/関数 | 入力 | 出力 | 役割 |
|---------|---|---|---|---|
| **main.py** | `main()` | args | - | 全体フロー制御 |
| **model/model1.py** | `Model1`, `train()` | config | 訓練済みモデル | モデル定義・訓練ループ |
| **model/Base.py** | `TransformerEncoder`, `MultiHeadAttention`, `FNclf`, `genX()` | X (テンソル) | h (学習済み表現) | Transformer実装 |
| **Data.py** | `FakenewsDataset` | walk_list | Dataset | データ管理 |
| **news_RandomWalk.py** | `rand_walk()` | graph | walk_list | ウォーク生成 |
| **build_graph.py** | `build_graph()` | CSV/Excel | *.pt | グラフ構築 |
| **preprocess.py** | `encode_texts_with_bert()` | テキスト | npy | 埋め込み生成 |
| **utils.py** | `load_data()`, `test_once()` | - | metrics | ユーティリティ |
| **config/HeteroSGT.json** | - | - | Config dict | 設定管理 |

---

**作成日**: 2026年9月25日  
**プロジェクト**: HeterSGT (Heterogeneous Subgraph Transformer for Fake News Detection)
