import argparse, sys, os
from datetime import datetime

sys.path.insert(0, "../python")
from vaxxer.classifier import VaxxerClassifier
from vaxxer.utils import json2space
from vaxxer.comet_utils import load_api_key

desc = """
Train vaxxer models for COVID related Twitter data.
"""
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("feature_dir", type=str, help="Specify feature directory")
parser.add_argument(
    "--neural_model",
    type=str,
    choices=[None, "classic", "lstm"],
    default=None,
    help="Select type of NeuralModel to classify with",
)
parser.add_argument("--device", type=str, default="cuda", help="Select GPU device")
parser.add_argument(
    "--num_trials", type=int, default=5, help="Specify number of trials"
)
parser.add_argument(
    "--num_instances", type=int, default=1, help="Specify number of instances"
)
parser.add_argument(
    "--metric",
    type=str,
    choices=["f1", "acc", "auc"],
    default="auc",
    help="Select metric to optimize",
)
parser.add_argument(
    "--direction",
    type=str,
    choices=["maximize", "minimize"],
    default="maximize",
    help="Select direction for the optimization",
)
parser.add_argument(
    "--algo",
    type=str,
    choices=["TPE", "RND", "GRID"],
    default="GRID",
    help="Select parameter search algorithm",
)
parser.add_argument(
    "--fix", nargs="+", default=[], help="Specify fixed feature components"
)
parser.add_argument(
    "--exclude", nargs="+", default=[], help="Specify excluded feature components"
)
parser.add_argument(
    "--parameters",
    type=str,
    default=None,
    help="Specify path for parameters.json to set the search space",
)
parser.add_argument(
    "--experiment_folder", type=str, default=None, help="Specify experiment folder name"
)
parser.add_argument(
    "--comet_file",
    type=str,
    default="None",
    help="Specify file path containing comet.ml API_KEY",
)
parser.add_argument(
    "--comet_workspace",
    type=str,
    default="covid-vaccine",
    help="Specify Comet ML workspace name",
)
parser.add_argument(
    "--comet_project",
    type=str,
    default="model-performance-revision",
    help="Specify Comet ML project name",
)
# parser.add_argument("--comet_project", type=str, default="model-performance-significance", help="Specify Comet ML project name")
parser.add_argument(
    "--random_sample",
    action="store_true",
    help="Turn on to run models for 100 random samples",
)

args = parser.parse_args()

# initialize vaxxer model
if args.comet_file == "None":
    comet_key = None
else:
    comet_key = load_api_key(args.comet_file)

clf = VaxxerClassifier(args.feature_dir, args.neural_model, args.device, comet_key)
clf.set_components(args.fix, args.exclude)
clf.load_components()
# define search space
if args.parameters == None:
    parameters = [
        ("epochs", "fixed", 5),
        ("batches", "fixed", 256),
        ("dropout", "fixed", 0.1),
        ("inter_dim", "fixed", 128),
        ("lr_rate", "fixed", 0.001),
        ("mode", "fixed", "inter"),
        ("scheduler", "fixed", "constant"),
    ]
    export_dir = None
else:
    if args.parameters.endswith("parameters.json"):
        parameters = json2space(args.parameters)
        if args.experiment_folder != None:
            experiment_folder = args.experiment_folder
        else:
            experiment_folder = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        export_dir = os.path.join(args.feature_dir, experiment_folder)
    else:
        raise ValueError("Provide valid path for 'parameters.json' file!")
print(export_dir)

if args.random_sample:
    parameters.append(("sample_seed", "choice", list(range(100))))
print(parameters)

# tune parameters
best_config = clf.tune(
    parameters,
    args.algo,
    args.num_trials,
    args.metric,
    args.direction,
    args.num_instances,
    args.comet_workspace,
    args.comet_project,
    export_dir=export_dir,
)
