import sys, os, shutil, pytest
from datetime import datetime

sys.path.insert(0, "%s/scripts" % os.getcwd())
sys.path.insert(0, "%s/python" % os.getcwd())
from vaxxer.features import *
from vaxxer.generator import FeatureGenerator
from vaxxer.bert_utils import BERTS
from vaxxer.generator import FeatureGenerator
from vaxxer.classifier import VaxxerClassifier
from node_embedding import preprocess
from graph_utils import calculate_network_centrality

data_dir = "%s/tests/sample_data" % os.getcwd()
seed_fp = os.path.join(data_dir, "seed_preprocessed", "valid_thread_seeds.csv")
label_fp = os.path.join(data_dir, "seed_labeled", "labeled_tweets.csv")
output_dir = os.path.join(data_dir, "components")
model_id = "Vax-skeptic_diFalse_tr0.70"

### generate components ###


def test_generate_bert():
    generator = FeatureGenerator(seed_fp, label_fp, output_dir, "Vax-skeptic", False)
    component = BertComponent(
        generator.tr_meta,
        generator.te_meta,
        generator.tr_text,
        generator.te_text,
        "ans/vaccinating-covid-tweets",
        120,
        False,
        False,
        normalize=False,
    )
    component.preprocess()
    generator.generate(component, {})
    assert "Labels" in os.listdir(os.path.join(output_dir, model_id))
    assert "Bert" in os.listdir(os.path.join(output_dir, model_id))
    assert (
        "norm:False_lemmatize:False_model:ans@vaccinating-covid-tweets_mtlen:120_stem:False"
        in os.listdir(os.path.join(output_dir, model_id, "Bert"))
    )


def test_generate_tfidf():
    generator = FeatureGenerator(seed_fp, label_fp, output_dir, "Vax-skeptic", False)
    component = TfIdfComponent(
        generator.tr_meta,
        generator.te_meta,
        generator.tr_text,
        generator.te_text,
        True,
        True,
        False,
        normalize=True,
    )
    component.preprocess()
    generator.generate(component, {"dimension": 100})
    assert "TfIdf" in os.listdir(os.path.join(output_dir, model_id))
    assert "norm:True_dimension:100_lemmatize:True_stem:True" in os.listdir(
        os.path.join(output_dir, model_id, "TfIdf")
    )


def test_generate_history():
    generator = FeatureGenerator(seed_fp, label_fp, output_dir, "Vax-skeptic", False)
    component = UserHistory(
        generator.tr_meta,
        generator.te_meta,
        generator.tr_text,
        generator.te_text,
        normalize=True,
    )
    component.preprocess()
    generator.generate(component, {"user_history": True})
    assert "History" in os.listdir(os.path.join(output_dir, model_id))
    assert "norm:True" in os.listdir(os.path.join(output_dir, model_id, "History"))


def test_generate_history_without_preprocess():
    with pytest.raises(RuntimeError):
        generator = FeatureGenerator(
            seed_fp, label_fp, output_dir, "Vax-skeptic", False
        )
        component = UserHistory(
            generator.tr_meta,
            generator.te_meta,
            generator.tr_text,
            generator.te_text,
            normalize=True,
        )
        # preprocess funciton is not called to raise RuntimeError
        generator.generate(component, {"user_history": True})


def test_generate_twitter():
    generator = FeatureGenerator(seed_fp, label_fp, output_dir, "Vax-skeptic", False)
    component = UserTwitter(generator.tr_meta, generator.te_meta, normalize=True)
    component.preprocess()
    generator.generate(component, {"user_twitter": True})
    assert "Twitter" in os.listdir(os.path.join(output_dir, model_id))
    assert "norm:True" in os.listdir(os.path.join(output_dir, model_id, "Twitter"))


def test_generate_centrality():
    graph_file = preprocess(data_dir, 1)
    centrality_file = calculate_network_centrality(graph_file)
    generator = FeatureGenerator(seed_fp, label_fp, output_dir, "Vax-skeptic", False)
    component = UserCentrality(
        generator.tr_meta, generator.te_meta, centrality_file, normalize=True
    )
    component.preprocess()
    generator.generate(component, {"user_centrality": True})
    assert "Centrality" in os.listdir(os.path.join(output_dir, model_id))
    assert "norm:True" in os.listdir(os.path.join(output_dir, model_id, "Centrality"))


def test_generate_node_embedding():
    generator = FeatureGenerator(seed_fp, label_fp, output_dir, "Vax-skeptic", False)
    node_emb_dir = os.path.join(data_dir, "node_embeddings")
    component = NodeEmbedding(
        generator.tr_meta,
        generator.te_meta,
        node_emb_dir,
        ["Walklets.csv"],
        normalize=True,
    )
    component.preprocess()
    generator.generate(component, {"user_ne": "Walklets.csv"})
    assert "Network" in os.listdir(os.path.join(output_dir, model_id))
    assert "norm:True_user_ne:Walklets.csv" in os.listdir(
        os.path.join(output_dir, model_id, "Network")
    )


### train classifier ###


def test_classifier():
    feature_dir = os.path.join(output_dir, model_id)
    clf = VaxxerClassifier(feature_dir, "lstm", None, None)
    clf.set_components([], [".ipynb_checkpoints"])
    clf.load_components()
    # define search space
    parameters = [
        ("epochs", "fixed", 5),
        ("batches", "fixed", 256),
        ("dropout", "fixed", 0.1),
        ("inter_dim", "fixed", 128),
        ("lr_rate", "fixed", 0.001),
        ("mode", "fixed", "inter"),
        ("scheduler", "fixed", "constant"),
    ]
    experiment_folder = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    export_dir = os.path.join(output_dir, experiment_folder)
    best_config = clf.tune(parameters, "GRID", 1, export_dir=export_dir)
    assert len(best_config) > 0
    assert experiment_folder in os.listdir(output_dir)


def test_classifier_no_text_feature():
    with pytest.raises(RuntimeError):
        feature_dir = os.path.join(output_dir, model_id)
        for folder_name in ["TfIdf", "Bert"]:
            text_dir = os.path.join(feature_dir, folder_name)
            if os.path.exists(text_dir):
                shutil.rmtree(text_dir)
        clf = VaxxerClassifier(feature_dir, "lstm", None, None)
        clf.load_components()


def test_classifier_no_components():
    with pytest.raises(RuntimeError):
        shutil.rmtree(output_dir)
        feature_dir = os.path.join(output_dir, model_id)
        clf = VaxxerClassifier(feature_dir, "lstm", None, None)
