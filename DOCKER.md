# Dockerでの実行

`compose.yaml` はリポジトリを `/workspace/HeterSGT`、その隣の `../Data` を `/workspace/Data` にマウントします。従来の `../Data/<dataset>/...` というパスを変更せずに利用できます。結果はホストの `results/` に保存されます。実行するユーザーIDを渡し、生成ファイルがホストで編集できるようにします。

## ビルドと確認

リポジトリ `/home/ogawa/HeterSGT` で実行します。初回ビルドではPythonパッケージとspaCyの英語モデルを取得するため、ネットワーク接続が必要です。Dockerビルド中に、Python 3.11で読み込めるようDeepWalk 1.0.3のインポート箇所だけを補正します。

```bash
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose build cpu
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose run --rm cpu python -c "import torch, spacy, torch_geometric; print(torch.__version__, torch.cuda.is_available()); spacy.load('en_core_web_sm')"
```

## ReCOVeryをCPUで実行

```bash
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose run --rm cpu python preprocess.py --dataset ReCOVery --hiddenSize 600 --device cpu
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose run --rm cpu python build_graph.py --dataset ReCOVery --hiddenSize 600
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose run --rm cpu python main.py --dataset ReCOVery --hiddenSize 600 --seed 7 --runs 10
```

既存の前処理済みグラフを再利用するなら、`preprocess.py` と `build_graph.py` は省略できます。前処理済みニュース特徴量を更新した場合は、グラフも再構築してください。

## GPUで実行

このホストにはNVIDIAのAPTリポジトリとGPUドライバがありますが、`nvidia-container-toolkit` が入っていません。管理者権限のあるホストのターミナルで、[NVIDIA公式の手順](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)に沿って次を一度実行します。Dockerサービスを再起動するため、実行前に稼働中のコンテナを確認してください。

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 nvidia-smi
```

最後のコマンドでGPU名が表示された後、以下の `gpu` サービスを使います。前処理では `--device cuda` を指定します。

```bash
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose run --rm gpu python -c "import torch; print(torch.cuda.is_available())"
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose run --rm gpu python preprocess.py --dataset ReCOVery --hiddenSize 600 --device cuda
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose run --rm gpu python build_graph.py --dataset ReCOVery --hiddenSize 600
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose run --rm gpu python main.py --dataset ReCOVery --hiddenSize 600 --seed 7 --runs 10
```

Toolkitの設定前は `docker run --gpus all` が `no known GPU vendor found` で失敗し、登録済みの `nvidia` ランタイムも実行ファイルがないため起動できません。CPUサービスはこの制約を受けません。

MC-Fakeは前処理の入力形式が未対応のため、Docker化だけでは実行できません。Dockerでもホストと同じ前処理コードを使用します。
