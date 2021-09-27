import sys, os
import numpy as np
script_path = os.path.realpath(__file__)
script_dir = os.path.split(script_path)[0]
sys.path.insert(0, '%s/../python' % script_dir)
from vaxxer.models import VaxxerClassifier
from sklearn.metrics import roc_auc_score

prepared_data_dir = "%s/../data/covid_vaxxer_representations/" % script_dir
classifier = VaxxerClassifier("tfidf", "Vax-skeptic", True)
config = {"model":"newton-cg"}

def load_and_evaluate(use_text, use_history, use_network):
    X_train, X_test = classifier.load(prepared_data_dir, use_text=use_text, use_history=use_history, use_network=use_network)
    model, tr_pred, te_pred = classifier.fit_vaxxer_classifier(X_train, X_test, config)
    auc_score = roc_auc_score(te_pred["label"], te_pred["proba"])
    return X_train, auc_score

def test_text_representation():
    X_train, auc_score = load_and_evaluate(True, False, False)
    assert X_train.shape[1] == 1000
    assert np.abs(0.8385-auc_score) < 0.0001
    
def test_text_history_representation():
    X_train, auc_score = load_and_evaluate(True, True, False)
    assert X_train.shape[1] == 1004
    assert np.abs(0.8743-auc_score) < 0.0001

def test_text_network_representation():
    X_train, auc_score = load_and_evaluate(True, False, True)
    assert X_train.shape[1] == 1128
    assert np.abs(0.9024-auc_score) < 0.0001
    
def test_text_history_network_representation():
    X_train, auc_score = load_and_evaluate(True, True, True)
    assert X_train.shape[1] == 1132
    assert np.abs(0.9110-auc_score) < 0.0001
