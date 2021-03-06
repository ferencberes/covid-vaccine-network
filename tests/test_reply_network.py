import sys, os, shutil
import pandas as pd
import networkx as nx
script_path = os.path.realpath(__file__)
script_dir = os.path.split(script_path)[0]
sys.path.insert(0, '%s/../python' % script_dir)
sys.path.insert(0, '%s/../scripts' % script_dir)
from graph_utils import get_user_graph, fit_embedding, EMBEDDINGS
from node_embedding import preprocess, fit

data_dir = "%s/sample_data" % script_dir
replies_fp = os.path.join(data_dir, "seed_preprocessed", "valid_threads.csv")
replies_df = pd.read_csv(replies_fp)

def test_original_graph_size():
    G = nx.from_pandas_edgelist(replies_df, source="usr_id_str", target="in_reply_to_user_id_str")
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 9

def test_graph_preprocessing_no_loop():
    G = get_user_graph(replies_df, 3, None, "usr_id_str", "in_reply_to_user_id_str")
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 6
    
def test_generate_embedding():
    G = get_user_graph(replies_df, 3, None, "usr_id_str", "in_reply_to_user_id_str")
    emb_df, _ = fit_embedding("DeepWalk", G, 10)
    assert emb_df.shape[0] == 4
    assert emb_df.shape[1] == 10
    
def test_all_embeddings():
    preproc_graph_path = preprocess(replies_fp, 1)
    for model in ["DeepWalk","NetMF","RandNE","SymmNMF","LaplacianEigenmaps","BigClam"]:
        output_fp, emb_df = fit(preproc_graph_path, model, 3)
        os.remove(output_fp)
        print(model)
        assert emb_df.shape[0] == 5
        assert emb_df.shape[1] == 3
            