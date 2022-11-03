import networkx as nx
import pandas as pd
import datetime as dt
import time

#####################
# Graph preprocessing
#####################

def create_graph_from_df(df: pd.DataFrame, src_col: str, trg_col: str):
    edges = list(zip(df[src_col],df[trg_col]))
    # filter out duplicates
    edges = list(set(edges))
    G = nx.Graph() 
    G.add_edges_from(edges)
    return G

def graph_min_degree(G, n_conn):
    return nx.k_core(G, k=n_conn)

def connected_component(G):
    C = max(nx.connected_components(G), key=len)
    return G.subgraph(C).copy()

def get_user_graph(reply_df: pd.DataFrame, n_conn: int, last_date_str: str=None, src_col: str="usr_id_str", trg_col: str="in_reply_to_user_id_str", remove_loops: bool=True, lcc_only=True):
    df = reply_df.copy()
    # excluding seed tweets
    df = df[~df[trg_col].isnull()]
    # setting proper type for node ids
    for col in [src_col, trg_col]:
        df[col] = df[col].astype("int64")
    print("seeds excluded", df.shape)
    if last_date_str is not None:
        df = df[df['date'] < last_date_str].copy()
        print("Edges were filtered", df.shape)
    if remove_loops:
        df = df[df[src_col]!=df[trg_col]]
        print("Loops were removed", df.shape)
    # creating the full user graph
    graph = create_graph_from_df(df, src_col, trg_col)
    N = graph.number_of_nodes()
    print("Original graph:", nx.info(graph))
    # keeping only the nodes with at least n connections
    if n_conn > 1:
        graph = graph_min_degree(graph, n_conn=n_conn)
        print("Graph after degree filter:", nx.info(graph))
    # choosing the largest connected component
    if lcc_only:
        graph = connected_component(graph)
        print("Largest connected component:", nx.info(graph))
    print("Kept node ratio: %.2f" % (graph.number_of_nodes()/N))
    return graph

#####################
# Node embeddings
#####################

def relabeled_nodes_labels(g: nx.Graph):
    # relabeling the nodes from 0 accordingly
    mapping = dict(zip(g, range(g.number_of_nodes())))
    g = nx.relabel_nodes(g, mapping)
    return g, mapping

from karateclub import Walklets, Role2Vec, Diff2Vec, DeepWalk, BoostNE, NodeSketch, NetMF, HOPE, GraRep, NMFADMM, GraphWave, LaplacianEigenmaps, SocioDim, RandNE, GLEE, Node2Vec
from karateclub import DANMF, NNSED, MNMF, BigClam, SymmNMF, GEMSEC

EMBEDDINGS = [
    'NetMF', 
    'DeepWalk',
    'Role2Vec', 
    'Diff2Vec', 
    'SocioDim', 
    'RandNE',
    'GLEE', 
    'NodeSketch', 
    'BoostNE', 
    'Walklets', 
    'GraRep', 
    'Node2Vec',
    'NMFADMM', 
    'LaplacianEigenmaps', 
    'GraphWave',
    'HOPE',
    'DANMF',
    'NNSED',
    'MNMF',
    'BigClam',
    'SymmNMF',
    'GEMSEC'
]

def karate_factory(algo: str, dim: int=128, nwalks: int=10, workers: int=4):
    if algo == "Walklets":
        karate_obj = Walklets(dimensions=int(dim/4), walk_number=nwalks, workers=workers)
    elif algo == "Role2Vec":
        karate_obj = Role2Vec(dimensions=dim, walk_number=nwalks, workers=workers)
    elif algo == "Diff2Vec":
        karate_obj = Diff2Vec(dimensions=dim, diffusion_number=nwalks, workers=workers)
    elif algo == "DeepWalk":
        karate_obj = DeepWalk(dimensions=dim, walk_number=nwalks, workers=workers)
    elif algo == "BoostNE":
        karate_obj = BoostNE(dimensions=int(dim/17)+1)
    elif algo == "NodeSketch":
        karate_obj = NodeSketch(dimensions=dim)
    elif algo == "NetMF":
        karate_obj = NetMF(dimensions=dim)
    elif algo == "HOPE":
        karate_obj = HOPE(dimensions=dim)
    elif algo == "GraRep":
        karate_obj = GraRep(dimensions=int(dim/5)+1)
    elif algo == "NMFADMM":
        karate_obj = NMFADMM(dimensions=int(dim/2))
    elif algo == "GraphWave":
        karate_obj = GraphWave()
    elif algo == "LaplacianEigenmaps":
        karate_obj = LaplacianEigenmaps(dimensions=dim)
    elif algo == "SocioDim":
        karate_obj = SocioDim(dimensions=dim)
    elif algo == "RandNE":
        karate_obj = RandNE(dimensions=dim)
    elif algo == "GLEE":
        karate_obj = GLEE(dimensions=dim)
    elif algo == "DANMF":
        karate_obj = DANMF(layers=[dim*4, dim])
    elif algo == "NNSED":
        karate_obj = NNSED(dimensions=dim)
    elif algo == "MNMF":
        karate_obj = MNMF(dimensions=dim)
    elif algo == "BigClam":
        karate_obj = BigClam(dimensions=dim)
    elif algo == "SymmNMF":
        karate_obj = SymmNMF(dimensions=dim)
    elif algo == "GEMSEC":
        karate_obj = GEMSEC(dimensions=dim, walk_number=nwalks)
    else:
        raise RuntimeError("Invalid model type: %s" % algo)
    return karate_obj

def fit_embedding(embedding_id: str, G: nx.Graph, dimension: int):
    print(nx.info(G))
    # renumbering the graph, and saving the numbers and mapping
    relabeled_graph, mapping = relabeled_nodes_labels(G)
    print("User graph renumbered.")
    start = time.time()
    model = karate_factory(embedding_id, dimension)
    model.fit(relabeled_graph)
    print("Fit process finished.")
    end = time.time()
    embedding_time = str(dt.timedelta(seconds=round(end - start)))
    print("Fitting time: ", embedding_time)
    node_embedding_vectors = model.get_embedding()
    print(len(mapping), len(node_embedding_vectors.shape))
    mapping_list = [(k, v) for k, v in mapping.items()]
    mapping_list.sort(key=lambda x: x[1])
    user_ids, _ = zip(*mapping_list)
    user_ids = [str(int(i)) for i in user_ids]
    embedding_df = pd.DataFrame(data=node_embedding_vectors, index=user_ids)
    return embedding_df, embedding_time

### Network centrality ###

def calculate_network_centrality(graph_file: str, src_col: str="usr_id_str", trg_col: str="in_reply_to_user_id_str",):
    edges_df = pd.read_csv(graph_file)
    G = nx.from_pandas_edgelist(edges_df, source=src_col, target=trg_col, create_using=nx.DiGraph)
    print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
    indeg_c = nx.in_degree_centrality(G)
    outdeg_c = nx.out_degree_centrality(G)
    kcore = nx.core_number(G)
    pr = nx.pagerank(G, max_iter=50)
    centralities = {
        "indeg":indeg_c,
        "outdeg":outdeg_c,
        "kcore":kcore,
        "pagerank":pr
    }
    centrality_df = pd.DataFrame(centralities)
    centrality_df = centrality_df.reset_index().rename({"index":src_col}, axis=1)
    centrality_file = graph_file.replace(".csv","_centrality.csv")
    centrality_df.to_csv(centrality_file, index=False)
    return centrality_file