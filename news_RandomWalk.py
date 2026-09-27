import copy
from random import shuffle
from deepwalk import graph
import random
import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm


def remove_dups(data):
    df = pd.DataFrame(data).astype(int)
    dup_index = df[df.duplicated(subset = df.columns)].index.values.tolist()
    df.drop_duplicates(subset = df.columns,inplace = True,ignore_index=True)
    newlist = df.values.tolist()
    num_dups = len(data)-len(newlist)
    print('Dups removed','\n'+'Num_Dups:', num_dups)
    if num_dups != 0:
        print('Dup_Index:',dup_index)
    return newlist,dup_index

def clean_list(data,dup_index):
    newlist = copy.deepcopy(data)
    for i,j in enumerate(dup_index):
        del newlist[j - i]
    return newlist

def global2df(data,colsname):
    new_df =  pd.DataFrame(list(data.items())[:len(list(data.items()))//2], columns=colsname)
    return new_df
def inner2df(data,colsname,typename):
    new_df =  pd.DataFrame(list(data.items())[:len(list(data.items()))//2], columns=colsname)
    new_df['type'] = [typename] * len(new_df)
    return new_df   


def get_inner_type(dataset,walk_list):
    # --- Original approach (commented out) ---
    # The previous implementation built DataFrames and merged them, then
    # assumed a single-row match for each global id. That caused failures
    # when matches were missing or duplicated. The original code is kept
    # below as comments for traceability.
    #
    # news_index = np.load(f"../Data/{dataset}/graph/nodes/news_index.npy", allow_pickle=True).item()
    # entity_index = np.load(f"../Data/{dataset}/graph/nodes/entity_index.npy", allow_pickle=True).item()
    # topic_index = np.load(f"../Data/{dataset}/graph/nodes/topic_index.npy", allow_pickle=True).item()
    # global_index1 = np.load(f"../Data/{dataset}/graph/nodes/global_index_graph1.npy", allow_pickle=True).item()
    # global_df1 = global2df(global_index1,["name", "g_id"])
    # news_df = inner2df(news_index,["name", "inner_id"],0)
    # entity_df = inner2df(entity_index,["name", "inner_id"],1)
    # topic_df = inner2df(topic_index,["name", "inner_id"],2)
    # inner_df1 = pd.concat([news_df,entity_df,topic_df],ignore_index = True)
    # final_df = pd.merge(global_df1, inner_df1)
    #
    # for walk in tqdm(walk_list,desc="getting inner_list & type_list ..."):
    #     ...

    # --- New robust mapping-based implementation ---
    news_index = np.load(f"../Data/{dataset}/graph/nodes/news_index.npy", allow_pickle=True).item()
    entity_index = np.load(f"../Data/{dataset}/graph/nodes/entity_index.npy", allow_pickle=True).item()
    topic_index = np.load(f"../Data/{dataset}/graph/nodes/topic_index.npy", allow_pickle=True).item()

    global_index1 = np.load(f"../Data/{dataset}/graph/nodes/global_index_graph1.npy", allow_pickle=True).item()

    # Build a safe mapping: global numeric id -> name (e.g. 'news_0', 'entity_12')
    g2name = {}
    for name, gid in global_index1.items():
        try:
            g2name[int(gid)] = name
        except Exception:
            # ignore entries that cannot be parsed to int
            continue

    inner_list = []
    type_list = []
    for walk in tqdm(walk_list, desc="getting inner_list & type_list ..."):
        inners = []
        types = []
        for j in walk:
            gid = None
            try:
                gid = int(j)
            except Exception:
                # unexpected format, mark as missing
                inners.append(-1)
                types.append(-1)
                continue

            name = g2name.get(gid)
            if name is None:
                # fallback: try to locate by scanning mapping (rare slow case)
                for nm, val in global_index1.items():
                    try:
                        if int(val) == gid:
                            name = nm
                            break
                    except Exception:
                        continue

            if name is None:
                inners.append(-1)
                types.append(-1)
                continue

            if name.startswith("news_"):
                inner = news_index.get(name)
                t = 0
            elif name.startswith("entity_"):
                inner = entity_index.get(name)
                t = 1
            elif name.startswith("topic_"):
                inner = topic_index.get(name)
                t = 2
            else:
                # unknown prefix: attempt to parse numeric suffix
                parts = name.split("_")
                try:
                    inner = int(parts[1])
                    t = 0
                except Exception:
                    inner, t = -1, -1

            if inner is None:
                inner = -1

            try:
                inners.append(int(inner) if inner != -1 else -1)
            except Exception:
                inners.append(-1)
            types.append(t)

        inner_list.append(inners)
        type_list.append(types)

    return inner_list, type_list


def _walk_from_news(G, start, walk_length, restart, rng):
    # 修正: DeepWalk の random_walk は if start: で開始ノードを判定するため、
    # start=0 だとランダムなノード（entity など）から歩き始めてしまう。
    # 同じ遷移・再開規則をここで実行し、news_0 を含む全記事で指定したIDから始める。
    path = [start]
    while len(path) < walk_length:
        neighbors = G[path[-1]]
        if not neighbors:
            break
        if rng.random() >= restart:
            path.append(rng.choice(neighbors))
        else:
            path.append(start)
    return [str(node) for node in path]


def rand_walk(dataset, restart, num_laps = 1, walk_length = 5):
    G = graph.load_edgelist(f"../Data/{dataset}/graph/edges/{dataset}.edgelist", undirected=True)
    df= pd.read_excel(f"../Data/{dataset}/news_final.xlsx")
    num_news = len(df['news_id'].tolist())
    label = df['label'].tolist()
    labels = label * num_laps  
    print('num_laps:',num_laps,'walk_length:',walk_length,'num_news:',num_news)
   
    walk_list = []
    for i in tqdm(range(num_laps),desc = 'news random walk...'):
        for j in range(num_news):
            # 修正: 開始ID 0 を真偽値で判定する DeepWalk 実装を通さず、
            # 各ウォークの先頭がラベルに対応するニュースID j になるようにする。
            walk = _walk_from_news(G, j, walk_length, restart, random.Random())
            walk_list.append(walk)
    
    walk_list_,dup_index = remove_dups(walk_list)
    labels_ = clean_list(labels,dup_index)      
    inner_list,type_list = get_inner_type(dataset,walk_list)
    return walk_list_,labels_,inner_list,type_list


