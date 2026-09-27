import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData
import argparse

    #-------------------各種 edge 情報を読み込む関数------------------------
def load_edge(dataset,node):
    if node == 'news':
        df = pd.read_csv(f'../Data/{dataset}/graph/edges/news2news.csv',sep=',',encoding='utf-8')
    else:
        df = pd.read_excel(f'../Data/{dataset}/graph/edges/news2{node}.xlsx')
    pair = df.values.tolist()
    news_index = np.load(f'../Data/{dataset}/graph/nodes/news_index.npy',allow_pickle= True).item()
    index_dict = np.load(f'../Data/{dataset}/graph/nodes/{node}_index.npy',allow_pickle= True).item()
    edges = []
    edges_ = []
    for i in pair:
        head = news_index[i[0]]
        tail = index_dict[str(i[1])]
        edge = [head, tail]
        edge_ = [tail, head]
        edges.append(edge)
        edges_.append(edge_) 
    return edges,edges_

def build_graph(dataset,hiddenSize):
    # この関数は、前処理で作成したニュース・エンティティ・トピックの特徴量と
    # 各種 edge 情報を読み込み、PyG の HeteroData オブジェクトとして組み立てる。
    # これにより、学習時に使える異種グラフ形式へ変換する。
    

    #----------------'３種のノードの特徴ベクトル'と'6種のエッジ'と'正解ラベル'をファイルからインポートし、変数に格納------------------------

    news_attr = np.load(f'../Data/{dataset}/graph/nodes/news_embeddings_{hiddenSize}_final.npy')   
    # news の特徴ベクトルを保存先ファイルから読み込む。
    # 形状は通常 (num_news, hidden_size) になる。num_news×hidden_size の ２次元配列。
    news_attr = torch.from_numpy(news_attr)
    # NumPy 配列を PyTorch のテンソルに変換する。NumPy配列：数学的な行列、テンソル：PyTorchでの多次元配列のこと。
    
    entity_attr = np.load(f'../Data/{dataset}/graph/nodes/entity_embeddings_{hiddenSize}.npy')
    # entity の特徴ベクトルを読み込む。
    entity_attr = torch.from_numpy(entity_attr)
    # entity_attr を PyTorch のテンソルへ変換する。
    
    topic_attr = np.load(f'../Data/{dataset}/graph/nodes/topic_embeddings_{hiddenSize}.npy')
    # topic の特徴ベクトルを読み込む。
    topic_attr = torch.from_numpy(topic_attr)
    # topic_attr をテンソルに変換する。

    news2entity, news2entity_ = load_edge(dataset,'entity')
    # news と entity の関係 edge を取得する。
    # news2entity は forward 方向、news2entity_ は reverse 方向の edge。
   
    news2topic, news2topic_ = load_edge(dataset,'topic')
    # news と topic の関係 edge を取得する。
    # 正方向と逆方向の両方を準備する。

    news2news, news2news_ = load_edge(dataset,'news')
    # news 同士の関係 edge を取得する。
    # これはニュース間のリンク構造を表す。
    
    df_news = pd.read_excel(f'../Data/{dataset}/news_final.xlsx')
    # 学習時に使うニュースのラベル情報が入っている Excel を読み込む。
    label = df_news['label'].tolist()
    # label 列だけをリストにして取り出す。
    

    #----------------PyG の HeteroData オブジェクトを生成し、ノード・エッジ・ラベルを設定-----------------------------------

    data = HeteroData()
    # PyG の異種グラフ用オブジェクトを生成する。

    data['news'].x = news_attr
    # news ノードの特徴量を設定する。
    data['entity'].x = entity_attr
    # entity ノードの特徴量を設定する。
    data['topic'].x = topic_attr
    # topic ノードの特徴量を設定する。
 
    data['news', 'has', 'entity'].edge_index = torch.tensor(news2entity, dtype=torch.long).t().contiguous()
    # news から entity への edge を relation name 'has' で登録する。
    data['entity', 'has_1', 'news'].edge_index = torch.tensor(news2entity_, dtype=torch.long).t().contiguous()
    # entity から news への逆方向 edge を relation name 'has_1' で登録する。

    data['news', 'belongs', 'topic'].edge_index = torch.tensor(news2topic, dtype=torch.long).t().contiguous()
    # news から topic への edge を 'belongs' として登録する。
    data['topic', 'belongs_1', 'news'].edge_index = torch.tensor(news2topic_, dtype=torch.long).t().contiguous()
    # topic から news への逆方向 edge を登録する。
    
    data['news', 'links', 'news'].edge_index = torch.tensor(news2news, dtype=torch.long).t().contiguous()
    # news 同士の関連 edge を 'links' として登録する。
    data['news', 'links_', 'news'].edge_index = torch.tensor(news2news_, dtype=torch.long).t().contiguous()
    # news 同士の逆方向 edge も登録する。

    data['news'].y = torch.tensor(label,dtype = torch.long)
    # news ノードに対応する正解ラベルを設定する。
    # この y が最終的な分類対象になる。
    
    #----------------HeteroData の内容を確認し、ファイルに保存する-----------------------------------
    print('='*60)
    # グラフ全体の情報を表示する前の区切り線を出す。
    print('HeteroGraph:',dataset,'\n',data)
    # 生成した HeteroData の内容を画面に表示する。
    print(' num_nodes:',data.num_nodes,'\n','num_edges:',data.num_edges,'\n','Data has isolated nodes:',data.has_isolated_nodes(),'\n','Data is undirected:',data.is_undirected())
    # ノード数、エッジ数、孤立ノード有無、無向グラフかどうかを表示する。
    print('='*60,'\n')
    # 区切り線を閉じる。
    torch.save(data,f'../Data/{dataset}/graph/{dataset}_{hiddenSize}_final.pt')
    # 最終的な異種グラフを .pt ファイルとして保存する。

    return data
    # HeteroData オブジェクトを返して、呼び出し元で利用できるようにする。

    
    #--------edegelist のローカルIDをグローバルIDに変換する関数-------------------
def class2global(edgelist,global_index,classindex):
    indices_g = []
    for i in edgelist:
        ID = classindex[i]
        index_g = global_index[ID]
        indices_g.append(index_g)
    return indices_g


    #------HeteroData の内容を確認し、randomwalkで入力として使えるようにエッジリストを整形し、ファイルに保存する-----------------------------------

def get_edgeList(dataset,hiddenSize):
    # 対象データセットと埋め込みサイズを受け取り、random-walk 用の全体エッジリストを作る関数

    # 各ノードタイプ（news/entity/topic）のローカルインデックス辞書を読み込む
    news_index = np.load(f'../Data/{dataset}/graph/nodes/news_index.npy', allow_pickle=True).item()
    entity_index = np.load(f'../Data/{dataset}/graph/nodes/entity_index.npy', allow_pickle=True).item()
    topic_index = np.load(f'../Data/{dataset}/graph/nodes/topic_index.npy', allow_pickle=True).item()
    # 先ほど保存した HeteroData をファイルから読み込む（weights_only=False は完全データ読み込み）
    data = torch.load(f'../Data/{dataset}/graph/{dataset}_{hiddenSize}_final.pt', weights_only=False) #weights_only=Falseを追加

    # 保存時に追加した逆向きエッジを取り除く（random-walk 用には片方向の方が扱いやすいため）
    del data['entity', 'has_1', 'news']
    del data['topic', 'belongs_1', 'news']
    del data['news', 'links_', 'news']
    
    # news->entity の edge_index を取得し、先頭行（head list）と2行目（tail list）に分ける
    newsList0 = data['news', 'has', 'entity'].edge_index.tolist()[0]
    entityList = data['news', 'has', 'entity'].edge_index.tolist()[1]
    
    # news->topic の edge_index からそれぞれの head/tail リストを取得
    newsList1 = data['news', 'belongs', 'topic'].edge_index.tolist()[0]
    topicList = data['news', 'belongs', 'topic'].edge_index.tolist()[1]
    
    # news->news の links edge についても head/tail を取得（内部は同一ノードタイプ）
    news_List_h = data['news', 'links', 'news'].edge_index.tolist()[0]
    news_List_t = data['news', 'links', 'news'].edge_index.tolist()[1]
    
    # ノードクラス（news/entity/topic）のローカルIDを全体グラフのグローバルIDに変換するための辞書を読み込む
    global_index = np.load(f'../Data/{dataset}/graph/nodes/global_index_graph1.npy', allow_pickle=True).item()
   
    # 各クラスごとのローカルIDリストをグローバルIDリストへ変換する
    news0_g = class2global(newsList0,global_index,news_index)
    entity_g = class2global(entityList,global_index,entity_index)
    
    news1_g = class2global(newsList1,global_index,news_index)
    topic_g = class2global(topicList,global_index,topic_index)   
    
    news_h_g = class2global(news_List_h,global_index,news_index)
    news_t_g = class2global(news_List_t,global_index,news_index)
    
    # head 側ノード（news->entity / news->topic / news->news の head）を結合して1つの配列にする
    node_head = news0_g + news1_g + news_h_g
    # tail 側ノード（entity / topic / news の tail）を結合して1つの配列にする
    node_tail = entity_g + topic_g + news_t_g
    
    # global ID ペアをスペース区切りの文字列にして random-walk 用の edgelist 形式に整形する
    edgeList_rw = []
    for i in range(len(node_head)):
        head = node_head[i]
        tail = node_tail[i]
        edge_rw = str(head)+' '+str(tail)
        edgeList_rw.append(edge_rw)
    # ファイルへ書き出す（各行は "head tail" の形式）
    with open(f'../Data/{dataset}/graph/edges/{dataset}.edgelist','w',encoding = 'utf-8') as f:
        for i in edgeList_rw:
            f.write(str(i)+'\n')
        f.close()
    # 作成したエッジリスト文字列の配列を返す
    return edgeList_rw


if __name__ == "__main__":
    # このブロックは、このスクリプトが "python build_graph.py ..." として
    # 直接実行されたときにだけ動く入口処理です。
    # つまり、ファイルをimport しただけでは実行されません。
    
    parser = argparse.ArgumentParser(description='choose dataset & hiddenSize')
    # argparse を使ってコマンドライン引数を定義します。
    # --dataset: 対象データセット名を受け取る
    # --hiddenSize: ニュース埋め込み次元を受け取る
    parser.add_argument('--dataset', type=str, default='MM COVID')
    # --dataset 引数を追加し、デフォルト値は 'MM COVID' とします。
    parser.add_argument('--hiddenSize', type=int, default=600, help="news_emb_size")
    # --hiddenSize 引数を追加し、デフォルト値は 600 とします。
    # help には "news_emb_size" を表示します。

    args = parser.parse_args()
    # コマンドラインから受け取った引数を解析して args に格納します。

    dataset = args.dataset
    # 解析結果から dataset 名を取り出します。
    hiddenSize = args.hiddenSize
    # 解析結果から hiddenSize を取り出します。
    
    data_sum_graph = build_graph(dataset,hiddenSize)
    # build_graph() を呼び出して、HeteroData のグラフを生成し保存します。
    # 返り値は 全体デカgraph オブジェクトで、変数 data_sum_graph に保持されます。
    edgeList_rw = get_edgeList(dataset,hiddenSize)
    # get_edgeList() を呼び出して、random walk 用の edgelist を生成し保存します。
    # 返り値は edge list の配列です。
    
    print(f'graph & edgelist for {dataset} done') 
    # 最後に、指定した dataset に対して graph と edgelist の生成が完了したことを表示します。
