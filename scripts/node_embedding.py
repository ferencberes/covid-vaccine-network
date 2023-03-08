import argparse, sys, os, time
import datetime as dt
import networkx as nx
import pandas as pd

sys.path.insert(0, "../python")
from vaxxer.utils import original_data
from graph_utils import *


def log(message, filename="node_embedding_logfile.txt"):
    with open(filename, "a") as file:
        file.write("\n")
        file.write(message)


def preprocess(
    data_dir,
    min_connections,
    last_date=None,
    src_col="usr_id_str",
    trg_col="in_reply_to_user_id_str",
):
    # reading in data
    thread_fp = os.path.join(data_dir, "seed_preprocessed", "valid_threads.csv")
    start = time.time()
    thread_df, _ = original_data(thread_fp)  # , nrows=10000)
    print("Data reading finished: %i second" % int(time.time() - start))
    print("Dimensions:", thread_df.shape)
    # node embedding preprocessing, leaving out records after the specified last_date
    start = time.time()
    G = get_user_graph(
        thread_df, min_connections, last_date, src_col, trg_col, remove_loops=True
    )
    print("User graph created: %i seconds" % int(time.time() - start))
    edges_df = nx.to_pandas_edgelist(G, source=src_col, target=trg_col)
    fname = "%icore_%s.csv" % (min_connections, str(last_date))
    edges_df.to_csv(fname, index=False)
    return fname


def fit(
    preprocessed_path,
    embedding_id,
    dimension=128,
    output_dir=None,
    src_col="usr_id_str",
    trg_col="in_reply_to_user_id_str",
):
    edges_df = pd.read_csv(preprocessed_path)
    date_str = dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    G = nx.from_pandas_edgelist(
        edges_df, source=src_col, target=trg_col, create_using=nx.Graph()
    )
    df, embedding_time = fit_embedding(embedding_id, G, dimension)
    # save the embedding, only the users in the data will have embeddings
    # for users not in the data, using zero vectors are recommended
    filename = "%s_dim%i_%s.csv" % (embedding_id, dimension, date_str)
    if output_dir != None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, filename)
    else:
        output_path = filename
    df.to_csv(output_path, header=False, index=True)
    log(filename + "\t" + embedding_time)
    return output_path, df


def valid_date_type(arg_date_str):
    """custom argparse *date* type for user dates values given from the command line"""
    try:
        arg_date = dt.datetime.strptime(arg_date_str, "%Y-%m-%d")
        return arg_date.strftime("%Y-%m-%d")
    except ValueError:
        msg = "Given date ({0}) is not valid! Expected format, YYYY-MM-DD!".format(
            arg_date_str
        )
        raise argparse.ArgumentTypeError(msg)


# execute only if run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", help="Select action to perform")
    subparsers.required = True
    preproc_parser = subparsers.add_parser("preprocess")
    preproc_parser.add_argument("data_dir", type=str, help="Specify data directory")
    preproc_parser.add_argument(
        "--con",
        type=int,
        default=5,
        choices=range(2, 11),
        metavar="[2-10]",
        help="minimum connection of nodes (2-10) (default: 5)",
    )
    preproc_parser.add_argument(
        "--date",
        type=valid_date_type,
        default=None,
        metavar="2021-MM-DD",
        help="load data only before the specified date",
    )
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument(
        "preproc_path", type=str, help="Path for the preprocessed data"
    )
    fit_parser.add_argument(
        "--model",
        choices=EMBEDDINGS,
        type=str,
        default="RandNE",
        help="node embedding model (default: RandNE)",
    )
    fit_parser.add_argument(
        "--dim",
        type=int,
        default=128,
        choices=range(5, 257),
        metavar="[10-128]",
        help="choose the embedding dimension (10-128) (default: 128)",
    )
    fit_parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory"
    )
    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)
    if args.action == "preprocess":
        _ = preprocess(args.data_dir, args.con, args.date)
    else:
        _, _ = fit(args.preproc_path, args.model, args.dim, args.output_dir)
