import argparse
import random

import numpy as np
import torch
from torch_geometric import data
from torch.utils.data import Dataset, random_split, DataLoader
from model.model1 import Model1
import json
import os
from Data import Vocab, Data, FakenewsDataset
import news_RandomWalk as rw
from utils import load_data, test_once, print_results_once, save_results
from model.Base import genX


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class Config():
    def __init__(self):
        self.name = "model config"
    
    def print_config(self):
        for attr in self.attribute:
            print(attr)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='MM COVID')
    parser.add_argument('--hiddenSize', type=int, default=600)
    parser.add_argument('--config', type=str, default='./config/HeteroSGT.json', help='configuration file name.')
    parser.add_argument('--model', type=str, default='Model1')
    parser.add_argument('--num_laps', type=int, default=1, help="num_laps")
    parser.add_argument('--walk_length', type=int, default=5, help="walk length")
    parser.add_argument('--round', type=int, default=1, help='test round')
    parser.add_argument('--num_layers', type=int, default=5, help="num_layers")
    parser.add_argument('--case2', type=str, default="no", help="yes or no for case study II")
    parser.add_argument('--case3', type=str, default="h0", help="h0, mean, or max for case study III")
    parser.add_argument('--restart', type=float, default=0.1, help=" probability of restarts in rw")
    parser.add_argument('--topn', type=int, default=3, help=" num_topics each news linked to")
    args = parser.parse_args()
    return args

def train_with_validation(model, train_data, val_data, test_data, epochs, optimizer, device, args):
    """Choose model weights on validation data; evaluate test data once."""
    criterion = torch.nn.CrossEntropyLoss()
    train_x = genX(train_data, device)
    val_x = genX(val_data, device)
    train_labels = train_data.labels.to(device)
    val_labels = val_data.labels.to(device)

    best_f1 = float("-inf")
    best_epoch = None
    best_state = None
    best_val_result = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_scores, _, _, _ = model(train_x, device, args)
        loss = criterion(train_scores, train_labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scores, val_preds, _, _ = model(val_x, device, args)
            val_result, _ = test_once(val_preds, val_scores, val_labels)
        if val_result["test_f1_macro"] > best_f1:
            best_f1 = val_result["test_f1_macro"]
            best_epoch = epoch + 1
            best_val_result = val_result
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"Epoch {epoch + 1}/{epochs}: train loss={loss.item():.4f}, "
            f"validation macro-F1={val_result['test_f1_macro']:.4f}",
            flush=True,
        )

    model.load_state_dict(best_state)
    model.eval()
    test_x = genX(test_data, device)
    test_labels = test_data.labels.to(device)
    with torch.no_grad():
        test_scores, test_preds, _, _ = model(test_x, device, args)
        test_result, test_auc = test_once(test_preds, test_scores, test_labels)

    print(f"Best validation epoch: {best_epoch}")
    print_results_once(best_val_result, "validation")
    print_results_once(test_result, "test")
    save_results(args, test_result)
    auc_dir = os.path.join('results', args.model, 'auc')
    os.makedirs(auc_dir, exist_ok=True)
    torch.save(test_auc, os.path.join(auc_dir, f'{args.dataset}_r{args.round}_auc.pt'))


if __name__ == "__main__":
    # コマンドライン引数をパース(構文解析)して `args` に格納する
    args = arg_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed: {args.seed} (model and random walks; data split remains seed 0)")
    # 使用可能なら GPU（cuda）を使い、なければ CPU を使う設定を作る
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # `args` にデバイス情報を保持しておく（他の関数で参照するため）
    args.device = device
    # 実行時情報を一行で表示する（デバッグ・ログ用）
    print(f"Training Device: {device}, Dataset: {args.dataset},  Model : {args.model}, Test Round: {args.round}, num_laps: {args.num_laps}, walk_length: {args.walk_length}, hiddenSize: {args.hiddenSize},num_layers: {args.num_layers}")

    # 設定ファイル（JSON）を読み込み、データセットごとの Config オブジェクト群を作る
    with open(args.config, 'r') as f:
        config_dicts = json.load(f)
    configs = {}
    for config in config_dicts:
        # JSON の各エントリを Config インスタンスに詰め替える
        conf = Config()
        for key, value in config.items():
            setattr(conf, key, value)
        configs.update({
            config["dataset"] : conf
        })
    # コマンドラインで指定されたデータセット用の設定を選択する
    config = configs[args.dataset]
    
    
    # グラフと学習データ・テストデータを読み込む（utils.load_data が返す）
    graph, train_data, test_data = load_data(args)

    val_data = train_data.validation_data
    print(f'Walks: train={len(train_data)}, validation={len(val_data)}, test={len(test_data)}')

    # モデル設定に埋め込みサイズや正規化形状などを反映させる
    config.query_size, config.key_size, config.value_size = args.hiddenSize, args.hiddenSize, args.hiddenSize
    config.norm_shape = (args.walk_length, config.num_hiddens)
    # グラフ中のノード数を config に保存（各ノードタイプのサイズを把握するため）
    config.news_size = graph["news"].x.shape[0]
    if graph["entity"] != {}:
        config.entity_size = graph["entity"].x.shape[0]
    else:
        config.entity_size = 0 
    if graph["topic"] != {}:
        config.topic_size = graph["topic"].x.shape[0]
    else:
        config.topic_size = 0
    # ランダムウォーク長や隠れ層サイズなどを config にセット
    config.walk_length = args.walk_length
    config.num_hiddens = args.hiddenSize
    # args に drop-out 率をコピーしておく（他処理で使うため）
    args.dropout = config.dropout
    # 指定されたモデル名（文字列）からクラスを取り出してインスタンス化する
    model = globals()[args.model](config)
    # モデルを double precision（float64）に変換する
    model = model.double()

    # モデルを学習デバイスに移動（GPU または CPU）
    model = model.to(device)
    # オプティマイザ(モデルのパラメータを更新する役割)を作成（Adam）
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if not hasattr(args, 'epochs'):
        args.epochs = 100
    train_with_validation(
        model, train_data, val_data, test_data, args.epochs, optimizer, device, args,
    )
