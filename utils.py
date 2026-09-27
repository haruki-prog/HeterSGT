from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, roc_curve, auc
import torch
import math
from tqdm import tqdm
import time
import news_RandomWalk as rw
from Data import Vocab, Data, FakenewsDataset
import pandas as pd
import numpy as np
import os

def load_data(args):
    # 事前に保存した HeteroData グラフファイルを読み込む（CPU マップで安全にロード）
    #graph = torch.load(f"../Data/{args.dataset}/graph/{args.dataset}_{args.hiddenSize}_final.pt", map_location=torch.device('cpu'))
    graph = torch.load(
    f"../Data/{args.dataset}/graph/{args.dataset}_{args.hiddenSize}_final.pt",
    map_location=torch.device('cpu'),
    weights_only=False
    )
    # 読み込んだグラフの簡易表示（デバッグ目的）
    print(graph)
    # ランダムウォークを生成して、walk 列・ラベル・内部インデックス・タイプ列を取得する
    walk_list,labels,inner_list,type_list = rw.rand_walk(args.dataset, args.restart, args.num_laps, args.walk_length)
    # 訓練/検証分割の比率（ここでは固定で 0.1 = 10% を検証用にする）
    test_ratio = 0.1
    # FakenewsDataset のインスタンスを作成する（walk_list 等を渡す）
    dataset = FakenewsDataset(walk_list, inner_list, type_list, labels, graph)
    # dataset.get_train_id により、walk 単位で train と test の Dataset を返してもらう
    train_data, test_data = dataset.get_train_id(test_ratio)
    
    # グラフと分割済みデータセットを返す
    return graph, train_data, test_data

def test_once(preds, scores, y):
    # preds, scores, y はテンソルのまま渡されるので CPU 上に移して numpy 互換の操作を行いやすくする
    preds = preds.cpu()
    scores = scores.cpu()
    y = y.cpu()
    # 全体の正解率を計算
    test_acc = accuracy_score(y, preds)
    # --- Binary average 指標を計算 ---
    test_pre_b = precision_score(y, preds, average="binary")
    test_recall_b = recall_score(y, preds, average="binary")
    test_f1_b = f1_score(y, preds, average="binary")
    # AUC はクラス 1 に対する予測確率列を使って算出する
    test_auc_b  = roc_auc_score(y, scores[:,1])

    # --- Micro average 指標を計算 ---
    test_pre_micro = precision_score(y, preds, average="micro")
    test_recall_micro = recall_score(y, preds, average="micro")
    test_f1_micro = f1_score(y, preds, average="micro")
    test_auc_micro  = roc_auc_score(y, scores[:,1], average="micro")

    # --- Macro average 指標を計算 ---
    test_pre_macro = precision_score(y, preds, average="macro")
    test_recall_macro = recall_score(y, preds, average="macro")
    test_f1_macro = f1_score(y, preds, average="macro")
    test_auc_macro  = roc_auc_score(y, scores[:,1], average="macro")

    # ROC 曲線の点 (fpr, tpr) と閾値を取得し、AUC を計算する
    fpr, tpr, thresholds = roc_curve(y, scores[:,1])
    roc_auc = auc(fpr, tpr)
    auc_list = [fpr, tpr, thresholds, roc_auc]

    # 結果を辞書にまとめる（呼び出し側で参照しやすい形式）
    test_results = {"test_acc" : test_acc,
                "test_pre_b" : test_pre_b,
                "test_pre_micro" : test_pre_micro,
                "test_pre_macro" : test_pre_macro,
                "test_recall_b" : test_recall_b,
                "test_recall_micro" : test_recall_micro,
                "test_recall_macro" : test_recall_macro,
                "test_f1_b" : test_f1_b,
                "test_f1_micro" : test_f1_micro,
                "test_f1_macro" : test_f1_macro,
                "test_auc_b" : test_auc_b,
                "test_auc_micro" : test_auc_micro,
                "test_auc_macro" : test_auc_macro
                }
    # 指標辞書と ROC 関連配列を返す
    return test_results,auc_list


def print_results_once(train_result, stage="train"):
    # train_result の値をアンパックして可読な名前に割り当てる
    train_acc, train_pre_b, train_pre_micro, train_pre_macro, train_recall_b, train_recall_micro, train_recall_macro, train_f1_b, train_f1_micro, train_f1_macro, \
        train_auc_b, train_auc_micro, train_auc_macro = train_result.values()

    # フォーマットして標準出力へ表示（Binary / Micro / Macro の各平均でまとめて表示）
    print(f"Avg = Binary \n"
    f"{stage} Acc: {train_acc:.4f}, {stage} Pre: {train_pre_b:.4f}, {stage} Recall: {train_recall_b:.4f}, {stage} f1: {train_f1_b:.4f}, {stage} auc: {train_auc_b:.4f} \n"
    f"Avg = Micro \n"
    f"{stage} Acc: {train_acc:.4f}, {stage} Pre: {train_pre_micro:.4f}, {stage} Recall: {train_recall_micro:.4f}, {stage} f1: {train_f1_micro:.4f}, {stage} auc: {train_auc_micro:.4f} \n"
    f"Avg = Macro \n"
    f"{stage} Acc: {train_acc:.4f}, {stage} Pre: {train_pre_macro:.4f}, {stage} Recall: {train_recall_macro:.4f}, {stage} f1: {train_f1_macro:.4f}, {stage} auc: {train_auc_macro:.4f} \n"
    )

def save_results(args, train_result):
    # 結果辞書を DataFrame に変換（行一つ分として保存するためリストで包む）
    df = pd.DataFrame([train_result])
    # --- Original code (commented out) ---
    # df.to_excel(f"./results/{args.dataset}_{args.hiddenSize}_R{args.round}_WL{args.walk_length}_dp{args.dropout}_{args.num_layers}_layers_case2{args.case2}_case3{args.case3}_restart_{args.restart}_topn_{args.topn}.xlsx", index = False, encoding = 'utf-8')

    # --- Replacement: ensure parent directory exists and avoid unsupported 'encoding' kwarg ---
    # 出力ファイルパスを組み立てる（実験設定のパラメータをファイル名に含める）
    out_path = f"./results/{args.dataset}_{args.hiddenSize}_R{args.round}_WL{args.walk_length}_dp{args.dropout}_{args.num_layers}_layers_case2{args.case2}_case3{args.case3}_restart_{args.restart}_topn_{args.topn}.xlsx"
    # 親ディレクトリがなければ作成する（存在確認と作成）
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    except Exception:
        # ディレクトリ作成に失敗しても処理を止めない
        pass
    # 結果を Excel ファイルとして保存する
    df.to_excel(out_path, index = False)
