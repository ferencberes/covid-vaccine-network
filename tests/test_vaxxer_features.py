import sys, os, shutil
sys.path.insert(0, '../python')
from vaxxer.models import VaxxerModel, VaxxerClassifier

data_dir = "sample_data"
seed_fp = os.path.join(data_dir, "seed_preprocessed", "valid_thread_seeds.csv")
model_dir = os.path.join(data_dir, "seed_labeled")
comet_key = None
node_emb_dir=None

### text models ###

def test_text_tfidf():
    model = VaxxerModel(seed_fp, model_dir, "tfidf", "Vax-skeptic", True, comet_key, workers=1)
    parameters = [
        ("model", "fixed", 'newton-cg'),
        ("dimension", "choice", [1,5])
    ]
    best_config = model.tune_parameters(parameters, "GRID", num_trials=10, metric="auc", direction="maximize", train_ratio=0.5, export_metrics=False)
    assert "dimension" in best_config and not "model" in best_config and len(model.components) == 1
    
def test_text_word2vec():
    model = VaxxerModel(seed_fp, model_dir, "word2vec", "Vax-skeptic", False, comet_key, workers=1)
    parameters = [
        ("model", "fixed", 'newton-cg'),
        ("dimension", "fixed", 10),
        ("window", "fixed", 3),
        ("min_count", "fixed", 1),
        ("epochs", "int", [1,3]),
    ]
    best_config = model.tune_parameters(parameters, "TPE", num_trials=10, metric="auc", direction="maximize", train_ratio=0.5, export_metrics=False)
    assert "epochs" in best_config and not "dimension" in best_config and len(model.components) == 1

def test_text_doc2vec():
    model = VaxxerModel(seed_fp, model_dir, "doc2vec", "Vax-skeptic", True, comet_key, workers=1)
    parameters = [
        ("model", "fixed", 'newton-cg'),
        ("dimension", "fixed", 10),
        ("window", "fixed", 3),
        ("min_count", "fixed", 1),
        ("epochs", "int", [1,3]), 
    ]
    best_config = model.tune_parameters(parameters, "TPE", num_trials=10, metric="auc", direction="maximize", train_ratio=0.5, export_metrics=False)
    assert "epochs" in best_config and not "dimension" in best_config and len(model.components) == 1
    
### models with user statistics ###

def test_user_history():
    model = VaxxerModel(seed_fp, model_dir, "tfidf", "Vax-skeptic", True, comet_key, workers=1)
    config = {
        "model":'newton-cg',
        "dimension":5,
        "user_history":True,
    }
    X_train, X_test = model.get_train_test_embeddings(config, train_ratio=0.5)
    print(X_train.shape)
    assert len(model.components) == 2 and X_train.shape[1] == 9

def test_user_twitter():
    model = VaxxerModel(seed_fp, model_dir, "tfidf", "Vax-skeptic", True, comet_key, workers=1)
    config = {
        "model":'newton-cg',
        "dimension":5,
        "user_twitter":True,
    }
    X_train, X_test = model.get_train_test_embeddings(config, train_ratio=0.5)
    print(X_train.shape)
    assert len(model.components) == 2 and X_train.shape[1] == 9
    
def test_concatenation():
    model = VaxxerModel(seed_fp, model_dir, "tfidf", "Vax-skeptic", True, comet_key, workers=1)
    config = {
        "model":'newton-cg',
        "dimension":5,
        "user_history":True,
        "user_twitter":True,
    }
    X_train, X_test = model.get_train_test_embeddings(config, train_ratio=0.5)
    print(X_train.shape)
    assert len(model.components) == 3 and X_train.shape[1] == 13
    
### models with node embedding ###

def test_node_embedding():
    emb_dir = os.path.join(data_dir, "node_embeddings")
    model = VaxxerModel(seed_fp, model_dir, "tfidf", "Vax-skeptic", True, comet_key, workers=1, node_emb_dir=emb_dir)
    config = {
        "model":'newton-cg',
        "dimension":5,
        "user_history":True,
        "user_twitter":True,
        "user_ne":"Walklets.csv",
    }
    X_train, X_test = model.get_train_test_embeddings(config, train_ratio=0.5)
    print(X_train.shape)
    assert len(model.components) == 4 and X_train.shape[1] == (128+13)

### other model functionalities
    
def test_export_load_predict():
    emb_dir = os.path.join(data_dir, "node_embeddings")
    model = VaxxerModel(seed_fp, model_dir, "tfidf", "Vax-skeptic", True, comet_key, workers=1, node_emb_dir=emb_dir)
    config = {
        "model":'newton-cg',
        "dimension":5,
        "user_history":True,
        "user_ne":"Walklets.csv",
    }
    output_dir = model.export(os.path.join(data_dir, "exported_data"), config, train_ratio=0.5)
    # load exported data with classifier
    classifier = VaxxerClassifier("tfidf", "Vax-skeptic", True)
    X_train, _ = classifier.load(output_dir, use_text=True, use_history=False, use_network=False)
    assert X_train.shape[1] == 5
    X_train, _ = classifier.load(output_dir, use_text=True, use_history=True, use_network=False)
    assert X_train.shape[1] == 9
    X_train, X_test = classifier.load(output_dir, use_text=True, use_history=True, use_network=True)
    assert X_train.shape[1] == (9+128)
    model, tr_pred, te_pred = classifier.fit_vaxxer_classifier(X_train, X_test, config)
    assert model.solver == "newton-cg"
    
def test_deploy():
    deploy_dir = os.path.join(data_dir, "seed_labeled", "predictions")
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    emb_dir = os.path.join(data_dir, "node_embeddings")
    model = VaxxerModel(seed_fp, model_dir, "tfidf", "Vax-skeptic", True, comet_key, workers=1, node_emb_dir=emb_dir)
    config = {
        "model":'newton-cg',
        "dimension":5,
        "user_history":True,
        "user_twitter":True,
        "user_ne":"Walklets.csv",
    }
    model.deploy(config)
    assert os.path.exists(deploy_dir)