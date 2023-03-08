import sys, os, argparse

sys.path.insert(0, "../python")
from vaxxer.features import *
from vaxxer.generator import FeatureGenerator
from vaxxer.bert_utils import BERTS

desc = """
Tune or deploy vaxxer models for COVID related Twitter data.
"""
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("data_dir", type=str, help="Specify data directory")
parser.add_argument(
    "--output_folder", type=str, default="components", help="Specify data directory"
)
parser.add_argument(
    "--class_label",
    type=str,
    choices=["Multiclass", "Pro-vaxxer", "Irrelevant", "Vax-skeptic"],
    default="Vax-skeptic",
    help="Specify vaxxer label to train binary classifier",
)
parser.add_argument(
    "--drop_irrelevant", action="store_true", help="Remove irrelevant category"
)
parser.add_argument(
    "--normalize", action="store_true", help="Normalize feature component"
)
parser.add_argument(
    "--train_ratio",
    type=float,
    default=0.7,
    help="Define train ratio for labeled tweets",
)
parser.add_argument(
    "--tweet_filter", type=str, default=None, help="Load tweet ids for data filtering"
)

subparsers = parser.add_subparsers(
    dest="feature_component", help="Select feature component to generate"
)
subparsers.required = True

tfidf_parser = subparsers.add_parser("tfidf")
tfidf_parser.add_argument("--stem", action="store_true", help="Stem raw text")
tfidf_parser.add_argument("--lemmatize", action="store_true", help="Lemmatize raw text")
tfidf_parser.add_argument(
    "--only_emoji", action="store_true", help="Use only emoji tokens"
)
tfidf_parser.add_argument("--dimensions", nargs="+", default=["1000"])

bert_parser = subparsers.add_parser("bert")
bert_parser.add_argument("--stem", action="store_true", help="Stem raw text")
bert_parser.add_argument("--lemmatize", action="store_true", help="Lemmatize raw text")
bert_parser.add_argument(
    "--model",
    type=str,
    choices=list(BERTS.keys()),
    default="ans/vaccinating-covid-tweets",
    help="Select pre-trained bert model",
)
bert_parser.add_argument(
    "--max_tensor_len",
    type=int,
    default=120,
    help="Select tensor size for Bert tokenizer",
)

history_parser = subparsers.add_parser("history")

twitter_parser = subparsers.add_parser("twitter")

network_parser = subparsers.add_parser("network")
network_parser.add_argument(
    "--embeddings", nargs="+", default=["Walklets_dim128_2021-09-05_12:03:25.csv"]
)
network_parser.add_argument("--emb_dir", type=str, default="node_embeddings_3conn")

centrality_parser = subparsers.add_parser("centrality")
centrality_parser.add_argument("--core", type=int, default=3)

args = parser.parse_args()

seed_fp = os.path.join(args.data_dir, "seed_preprocessed", "valid_thread_seeds.csv")
label_fp = os.path.join(args.data_dir, "seed_labeled", "labeled_tweets.csv")
output_dir = os.path.join(args.data_dir, args.output_folder)
generator = FeatureGenerator(
    seed_fp,
    label_fp,
    output_dir,
    args.class_label,
    args.drop_irrelevant,
    train_ratio=args.train_ratio,
    tweet_filter=args.tweet_filter,
)

if args.feature_component == "tfidf":
    component = TfIdfComponent(
        generator.tr_meta,
        generator.te_meta,
        generator.tr_text,
        generator.te_text,
        args.stem,
        args.lemmatize,
        args.only_emoji,
        normalize=args.normalize,
    )
    param_list = [{"dimension": int(dim)} for dim in args.dimensions]
elif args.feature_component == "bert":
    component = BertComponent(
        generator.tr_meta,
        generator.te_meta,
        generator.tr_text,
        generator.te_text,
        args.model,
        args.max_tensor_len,
        args.stem,
        args.lemmatize,
        normalize=args.normalize,
    )
    param_list = [{}]
elif args.feature_component == "history":
    component = UserHistory(
        generator.tr_meta,
        generator.te_meta,
        generator.tr_label,
        generator.te_label,
        normalize=args.normalize,
    )
    param_list = [{"user_history": True}]
elif args.feature_component == "twitter":
    component = UserTwitter(
        generator.tr_meta, generator.te_meta, normalize=args.normalize
    )
    param_list = [{"user_twitter": True}]
elif args.feature_component == "network":
    node_emb_dir = os.path.join(args.data_dir, args.emb_dir)
    if "All" in args.embeddings:
        embedding_files = os.listdir(node_emb_dir)
        embedding_files.remove("logs")
    else:
        embedding_files = args.embeddings
    print(embedding_files)
    component = NodeEmbedding(
        generator.tr_meta,
        generator.te_meta,
        node_emb_dir,
        embedding_files,
        normalize=args.normalize,
    )
    param_list = [{"user_ne": emb} for emb in embedding_files]
elif args.feature_component == "centrality":
    centrality_file = "%s/reply_graphs/%icore_None_centrality.csv" % (
        args.data_dir,
        args.core,
    )
    component = UserCentrality(
        generator.tr_meta, generator.te_meta, centrality_file, normalize=args.normalize
    )
    param_list = [{"user_centrality": True}]
else:
    raise ValueError("Provide feature component type!")

component.preprocess()
for params in param_list:
    generator.generate(component, params)
print("done")
