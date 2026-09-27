# HeteroSGT セッション要約

## 目的と参照資料

論文 [HetroSGT.pdf](HetroSGT.pdf) に沿って前処理、グラフ構築、学習・評価を確認し、ReCOVery データセットで実行した。論文はニュース・エンティティ・トピックの異種グラフ、ニュース用の二段階 BiGRU＋注意機構、エンティティとトピック用の BERT、80%／10%／10% の訓練・検証・テスト分割を記述している。

## 実装と修正の経緯

- `preprocess.py`：ニュースの TF-IDF/SVD 特徴を単語・文の二段階 BiGRU＋注意機構に変更。エンティティは spaCy `en_core_web_sm` で抽出し、トピックは LDA で生成する。エンティティ名とトピック上位単語の特徴は BERT で初期化する。ReCOVery のトピック数は論文の設定に合わせて35とした。
- `preprocess.py`：ニュース埋め込みの教師あり学習に全記事のラベルを使っていたため、当初のテスト結果がほぼ1.0000となった。これはテストラベル漏洩と判断し、現在は `Data.py` と同じ訓練80%のラベルだけで BiGRU を学習する。検証・テスト記事の特徴量は生成するが、そのラベルは事前学習に使わない。漏洩したニュース埋め込みの再利用を防ぐため、`--skip_news_embeddings` は削除した。
- `preprocess.py`：`--device cuda` でニュース埋め込み、BERT、LDA を GPU で計算できるようにした。LDA は PyTorch の CUDA 実装を追加したため、scikit-learn 版とトピックが完全一致するとは限らない。エンティティ特徴を再利用する `--skip_entity_embeddings` は残っている。
- `Data.py`：ニュースIDを基準に80%／10%／10%へ分割する `get_train_val_test_id()` を追加。既存の `get_train_id()` は訓練・テストの2値を返し、検証データを `train_data.validation_data` に保持する。ReCOVery の2,029件では訓練1,623件、検証203件、テスト203件。
- `main.py`：100エポックの各回で訓練データを使い、検証 Macro-F1 が最高の重みを保持する。最後にその重みでテストデータを1回評価し、結果と AUC を保存する。旧方式のように毎エポックのテスト結果から最高値を選ぶ処理は使わない。
- `news_RandomWalk.py`：DeepWalk ライブラリが `start=0` を偽と判定してランダムなノードから歩き始める問題を修正。変更理由はコード内にコメントで記載した。実データの全2,029件について、ウォーク先頭が対応するニュースIDであることを確認済み。
- これまでの修正依頼では、指定されたファイル以外を変更しないという制約があった。今後 `preprocess.py` 以外のソースを修正する場合、具体的な修正点を明記するというユーザーの希望がある。

## 実行順

リポジトリ `/home/ogawa/HeterSGT` で実行する。入力CSVは `../Data/ReCOVery/recovery-news-data.csv`。

```bash
venv311/bin/python preprocess.py --dataset ReCOVery --device cuda --skip_entity_embeddings
venv311/bin/python build_graph.py --dataset ReCOVery --hiddenSize 600
venv311/bin/python main.py --dataset ReCOVery --hiddenSize 600
```

ニュース埋め込みを更新した場合は、`build_graph.py` も再実行する。ランダムウォークの修正だけならグラフの再生成は不要。コマンド末尾に余分な `~` を付けると `--hiddenSize` の整数解析に失敗する。

## 実行結果の読み方

提示された直近の実行では、`Best validation epoch: 36`、検証 Macro-F1 は **0.9194**、その36エポック目の重みで1回だけ測ったテスト Macro-F1 は **0.8733**、テスト AUC は **0.9480**。Binary／Micro／Macro はクラス間の集約方式であり、エポック間の平均ではない。

旧方式では同じテスト集合を100エポックで100回評価し、その中の最高テスト Macro-F1 を報告していた。これはテスト集合をモデル選択にも使うため、現在は検証集合で選び、テストは最後に評価する。

## 残る論点

- `main.py` には `--seed` 引数があるが、現状では学習・ランダムウォーク全体の乱数固定には使われていない。データ分割は `random_state=0`、ニュース埋め込みの事前学習は `torch.manual_seed(0)` で固定されている。
- 論文の「異なる乱数シードで10回試行し、テスト指標の平均±標準偏差を報告する」形式は**未実装**。分割を固定して10回行う方式なら主に `main.py` と `news_RandomWalk.py` の修正が必要。分割も各回で変えるなら、ラベル漏洩を避けるため `Data.py`、`preprocess.py`、`utils.py` も連動させ、各回の訓練集合だけでニュース埋め込みを再学習する必要がある。
- ReCOVery では出版社55種類がそれぞれ単一ラベルに対応しており、データの偏りに注意が必要。
