import sys, os, time, optuna, joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz
from datetime import datetime as dt
from optuna.samplers import TPESampler, RandomSampler, GridSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

sys.path.insert(0, "../")
from comet_utils import init_experiment, load_api_key

from .utils import *
from .features import *

MODELS = dict([(solver,LogisticRegression(solver=solver)) for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']])

def suggest_config(parameters, trial):
    """Convert manually defined search parameters into Optuna supported search spaces"""
    config = {}
    for c_name, c_type, c_vals in parameters:
        if c_type == "choice":
            config[c_name] = trial.suggest_categorical(c_name, c_vals)
        elif c_type == "int":
            config[c_name] = trial.suggest_int(c_name, c_vals[0], c_vals[1], step=c_vals[2] if len(c_vals) > 2 else 1, log=c_vals[3] if len(c_vals) > 3 else False)            
        elif c_type == "float":
            config[c_name] = trial.suggest_float(c_name, c_vals[0], c_vals[1], step=c_vals[2] if len(c_vals) > 2 else 1, log=c_vals[3] if len(c_vals) > 3 else False)            
        elif c_type == "fixed":
            config[c_name] = c_vals
        else:
            raise ValueError("Parameter type '%s' was not implemented!" % c_type)
    return config

def extract_grid(parameters):
    """Extract grid space from manually defined search parameters"""
    grid = {}
    for c_name, c_type, c_vals in parameters:
        if c_type == "choice":
            grid[c_name] = c_vals
        elif c_type == "fixed":
            grid[c_name] = [c_vals]
        else:
            raise ValueError("GridSearch can only use categorical search space!")
    return grid

def log_fixed_params(parameters, exp):
    """Utility function for Comet.ml logging"""
    for c_name, c_type, c_vals in parameters:
        if c_type == "fixed":
            exp.log_parameter(c_name, c_vals)
            
class VaxxerClassifier():
    def __init__(self, embedding_type="tfidf", class_label="Vax-skeptic", drop_irrelevant=True, verbose=False):
        if embedding_type in ["None", "tfidf", "doc2vec", "word2vec"]:
            self.embedding_type = embedding_type
        else:
            raise ValueError("Invalid 'embedding_type': %s! Use 'tfidf', 'doc2vec', 'word2vec'." % embedding_type)
        self.label_indexer = LabelIndexer()
        if class_label in ["Multiclass"] + self.label_indexer.labels:
            self.class_label = class_label
        else:
            raise ValueError("Choose 'Multiclass' or from %s" % str(self.label_indexer.labels))
        self.drop_irrelevant = drop_irrelevant
        self.verbose = verbose
        self._exported_meta_columns = ["id_str","usr_id_str","epoch","date"]
        
    def _clear_cache(self):
        self.tr_meta = None
        self.te_meta = None
        self.tr_label = None
        self.te_label = None
        self.components = []
        
    def load(self, model_dir, use_text=True, use_history=True, use_network=True, delimiter=","):
        """Load features, labels and other meta data."""
        self._clear_cache()
        # TODO: load parameters from filename!!!
        train_parts = []
        test_parts = []
        #load text feature matrix
        if use_text:
            tr_text = load_npz(os.path.join(model_dir, "train_text.npz"))
            te_text = load_npz(os.path.join(model_dir, "test_text.npz"))
            train_parts.append(tr_text.toarray())
            test_parts.append(te_text.toarray())
            print("text", tr_text.shape, te_text.shape)
        #load history feature matrix
        if use_history:
            tr_history = np.loadtxt(os.path.join(model_dir, "train_history.csv"), delimiter=delimiter)
            te_history = np.loadtxt(os.path.join(model_dir, "test_history.csv"), delimiter=delimiter)
            train_parts.append(tr_history)
            test_parts.append(te_history)
            print("history", tr_history.shape, te_history.shape)
        #load node embeddings
        if use_network:
            tr_network = np.loadtxt(os.path.join(model_dir, "train_network.csv"), delimiter=delimiter)
            te_network = np.loadtxt(os.path.join(model_dir, "test_network.csv"), delimiter=delimiter)
            train_parts.append(tr_network)
            test_parts.append(te_network)
            print("network", tr_network.shape, te_network.shape)
        #concatenation
        X_tr = np.concatenate(train_parts, axis=1)
        X_te = np.concatenate(test_parts, axis=1)
        print("After concatenation:", X_tr.shape, X_te.shape)
        #load labels
        self.tr_label = np.loadtxt(os.path.join(model_dir, "train_label.csv"), delimiter=delimiter)
        self.te_label = np.loadtxt(os.path.join(model_dir, "test_label.csv"), delimiter=delimiter)
        assert len(self.tr_label) == len(X_tr)
        assert len(self.te_label) == len(X_te)
        #load meta
        self.tr_meta = pd.read_csv(os.path.join(model_dir, "train_meta.csv"), delimiter=delimiter)
        self.te_meta = pd.read_csv(os.path.join(model_dir, "test_meta.csv"), delimiter=delimiter)
        assert len(self.tr_meta) == len(X_tr)
        assert len(self.tr_meta) == len(X_tr)
        return X_tr, X_te
    
    @property
    def model_id(self):
        """Generate string identifier for the model"""
        date_str = dt.now().strftime("%Y-%m-%d_%H:%M")
        return "%s_%s_%s_%s" % (date_str, self.embedding_type, self.class_label, self.drop_irrelevant)
    
    def fit_vaxxer_classifier(self, X_train, X_test, config):
        """Fit vaxxer classifier model."""
        if "model" in config:
            model = MODELS[config["model"]]
            model.fit(X_train, self.tr_label)
            tr_pred = self.tr_meta.copy()
            te_pred = self.te_meta.copy()
            tr_pred["proba"] = model.predict_proba(X_train)[:,1]
            te_pred["proba"] = model.predict_proba(X_test)[:,1]
            tr_pred["label"] = self.tr_label
            te_pred["label"] = self.te_label
        else:
            model, tr_pred, te_pred = None, None, None
        return model, tr_pred, te_pred

class VaxxerModel(VaxxerClassifier):
    def __init__(self, seed_fp, model_dir, embedding_type="None", class_label="Multiclass", drop_irrelevant=False, comet_key=None, meta_cols=['id_str', 'usr_id_str', 'epoch', 'date', 'usr_followers_count', 'usr_friends_count', 'usr_favourites_count', 'usr_statuses_count'], workers=2, node_emb_dir=None, verbose=True):
        super(VaxxerModel, self).__init__(embedding_type, class_label, drop_irrelevant, verbose)
        self.seed_fp = seed_fp
        self.model_dir = model_dir
        self.node_emb_dir = node_emb_dir
        self.label_fp = os.path.join(model_dir, "labeled_tweets.csv")
        self.comet_key = comet_key
        self.meta_cols = meta_cols
        self.workers = workers
        
    def _generate_embeddings(self, config):
        """Generate tweet representations based on the selected 'embedding_type'"""        
        tr_parts = []
        te_parts = []
        all_columns = []
        for comp in self.components:
            tr_tmp, te_tmp, cols = comp.generate(config)
            if cols != None:
                print(tr_tmp.shape,te_tmp.shape)
                tr_parts.append(tr_tmp)
                te_parts.append(te_tmp)
                all_columns += cols
        X_train = np.concatenate(tr_parts, axis=1)
        X_test = np.concatenate(te_parts, axis=1)
        print("Concatenated size:", X_train.shape, X_test.shape)
        self.feature_columns = all_columns
        return X_train, X_test
    
    def _run_single_config(self, train_ratio, config):
        """Run single experiment based on the parameters passed in the 'config' dictionary"""
        X_train, X_test = self._generate_embeddings(config)
        model_str = config["model"]
        model_tuples = [(model_str, MODELS[model_str])]
        is_multi = (self.class_label == "Multiclass")
        metrics_df = evaluate_classifiers(model_tuples, X_train, self.tr_label, X_test, self.te_label, multiclass=is_multi, show_confusion_matrix=True, verbose=False)
        metrics_df["class"] = self.class_label
        #append parameter values to dataframe
        for key, value in config.items():
            metrics_df[key] = value
        return metrics_df
    
    def _transform_labels(self, tr_label, te_label):
        if self.class_label == "Multiclass":
            self.tr_label = tr_label
            self.te_label = te_label
        else:
            label_id = self.label_indexer.label2idx(self.class_label)
            self.tr_label = (np.array(tr_label) == label_id).astype('int')
            if te_label is None:
                self.te_label = None
            else:
                self.te_label = (np.array(te_label) == label_id).astype('int')
                
    def _prepare_feature_components(self, tr_text, te_text, parameters):
        """Prepare feature components and execute preprocessing"""
        if self.embedding_type == "tfidf":
            self.components.append(TfIdfComponent(self.tr_meta, self.te_meta, tr_text, te_text))
        elif self.embedding_type in ["word2vec","doc2vec"]:
            self.components.append(GensimComponent(self.tr_meta, self.te_meta, tr_text, te_text, model_type=self.embedding_type, workers=self.workers))
        def extract_parameter_ids(parameters):
            if isinstance(parameters, list):
                param_dict = dict([(c_name,c_vals) for c_name, c_type, c_vals in parameters])
            else:
                param_dict = parameters.copy()
            return list(param_dict.keys()), param_dict.get("user_ne", "None")
        param_ids, node_emb_files = extract_parameter_ids(parameters)
        if "user_history" in param_ids:
            self.components.append(UserHistory(self.tr_meta, self.te_meta, self.tr_label, self.te_label))
        if "user_ne" in param_ids and node_emb_files != "None":
            if not isinstance(node_emb_files, list):
                node_emb_files = [node_emb_files]
            self.components.append(NodeEmbedding(self.tr_meta, self.te_meta, self.node_emb_dir, file_names=node_emb_files))
        if "user_twitter" in param_ids:
            self.components.append(UserTwitter(self.tr_meta, self.te_meta))
        for comp in self.components:
            comp.preprocess()
        
    def tune_parameters(self, parameters, search_alg, num_trials=5, metric="f1", direction="maximize", train_ratio=0.7, num_times=1, export_metrics=True):
        """Evaluate performance for multiple dimension parameter values"""
        self._clear_cache()
        model_id = self.model_id
        if self.comet_key != None:
            exp = init_experiment(self.comet_key, "model-performance", "covid-vaccine")
            exp.log_parameters({
                "model_id":model_id,
                "model_type":self.embedding_type,
                "multiclass":self.class_label,
                "train_ratio":train_ratio,
                "num_samples":num_trials,
                "metric":metric,
                "direction":direction,
                "search_alg":search_alg
            })
            log_fixed_params(parameters, exp)
            exp.add_tag("multi" if self.class_label == "Multiclass" else "binary")
        start = time.time()
        tr_text, tr_label, self.tr_meta, te_text, te_label, self.te_meta, _ = get_train_test_data(self.seed_fp, self.label_fp, train_ratio=train_ratio, meta_cols=self.meta_cols, drop_irrelevant=self.drop_irrelevant, visualize=False, verbose=self.verbose)
        self._transform_labels(tr_label, te_label)
        print("data loading:", time.time() - start, "seconds\n")
        start = time.time()
        self._prepare_feature_components(tr_text, te_text, parameters)
        print("total preprocessing:", time.time() - start, "seconds\n")
        metric_df_parts = []
        def objective(trial):
            config = suggest_config(parameters, trial)
            instances = []
            for _ in range(num_times):
                instance_df = self._run_single_config(train_ratio, config)
                instance_df = instance_df[instance_df["part"] == "test"]
                instances.append(instance_df)
            tmp_df = pd.concat(instances, axis=0)
            print("metrics 1", tmp_df.shape)
            group_cols = list(tmp_df.drop("score", axis=1).columns)
            print(group_cols)
            tmp_df = tmp_df.groupby(group_cols)["score"].agg(["mean","std"]).reset_index()
            print("metrics 2", tmp_df.shape)
            metric_df_parts.append(tmp_df)
            metrics = dict(zip(tmp_df["metric"],tmp_df["mean"]))
            return metrics[metric]
        if search_alg == "GRID":
            algo = GridSampler(extract_grid(parameters))
        elif search_alg == "RND":
            algo = RandomSampler()
        elif search_alg == "TPE":
            algo = TPESampler(n_startup_trials=int(num_trials*0.3))
        else:#default optuna setting
            algo = None
        study = optuna.create_study(direction="maximize", sampler=algo)
        study.optimize(objective, n_trials=num_trials, n_jobs=1)
        metrics_df = pd.concat(metric_df_parts)
        best_config = study.best_params
        print("Best config: ", best_config)
        if export_metrics:
            result_dir = os.path.join(self.model_dir, "results")
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            study_fp = os.path.join(result_dir, "%s.pkl" % model_id)
            print("Study file:", study_fp)
            joblib.dump(study, study_fp)
            result_fp = os.path.join(result_dir, "%s.csv" % model_id)
            print("Output file:", result_fp)
            metrics_df.to_csv(result_fp, index=False)
        if self.comet_key != None:
            exp.log_parameters(best_config)
            exp.log_metrics({
                "train_size":len(tr_text),
                "test_size":len(te_text)
            })
            best_results = dict(metrics_df.groupby("metric")["mean"].max()[["f1","acc","auc"]])
            exp.log_metrics(best_results)
            exp.end()
        return best_config
    
    def deploy(self, config, k=None):
        """Deploy model for the pre-defined dimension values based on 'k' random tweet samples."""
        pass
        self._clear_cache()
        model_id = self.model_id
        if self.comet_key != None:
            exp = init_experiment(self.comet_key, "model-deploy", "covid-vaccine")
            exp.log_parameters({
                "model_id":model_id,
                "model_type":self.embedding_type,
                "multiclass":self.class_label,
                "k":k,
            })
            exp.log_parameters(config)
            exp.add_tag("multi" if self.class_label == "Multiclass" else "binary")
        # load data
        start = time.time()
        tr_text, tr_label, self.tr_meta, _, _, _, unlabeled_df = get_train_test_data(self.seed_fp, self.label_fp, train_ratio=1.0, meta_cols=self.meta_cols, drop_irrelevant=self.drop_irrelevant, visualize=False, verbose=self.verbose)
        self.te_meta = unlabeled_df[self.meta_cols]
        te_text = unlabeled_df["full_text"].values
        self._transform_labels(tr_label, None)
        print("data loading:", time.time() - start, "seconds\n")
        # sample data
        if k!= None:
            unlabeled_df = unlabeled_df.sample(k)
        # preprocessing
        start = time.time()
        self._prepare_feature_components(tr_text, te_text, config)
        print("total preprocessing:", time.time() - start, "seconds\n")
        X_train, X_unkown = self._generate_embeddings(config)
        # fit model
        model = MODELS[config["model"]]
        model.fit(X_train, self.tr_label)
        # predict
        probas = model.predict_proba(X_unkown)
        predictions_df = unlabeled_df[["id_str","usr_id_str","epoch","date","full_text"]].copy()
        if self.class_label == "Multiclass":
            for idx, label in self.label_indexer.get_index_mapping().items():
                predictions_df[label] = probas[:,idx]
        else:
            predictions_df[self.class_label] = probas[:,1]
        self.predictions_df = predictions_df
        prediction_dir = os.path.join(self.model_dir, "predictions")
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
        self.predictions_df.to_csv(os.path.join(prediction_dir, "%s_k%s.csv" % (model_id, str(k))), index=False)
        if self.comet_key != None:
            exp.log_metrics({
                "labeled_size":len(tr_text),
                "unlabeled_size":len(unlabeled_df)
            })
            exp.end()
            
    def prepare_train_test_components(self, config, train_ratio=0.7):
        """Split data for train-test parts. Then, prepare different feature components (text, history, network) for the classification task."""
        self._clear_cache()
        start = time.time()
        tr_text, tr_label, self.tr_meta, te_text, te_label, self.te_meta, _ = get_train_test_data(self.seed_fp, self.label_fp, train_ratio=train_ratio, meta_cols=self.meta_cols, drop_irrelevant=self.drop_irrelevant, visualize=False, verbose=self.verbose)
        self._transform_labels(tr_label, te_label)
        print("data loading:", time.time() - start, "seconds\n")
        start = time.time()
        self._prepare_feature_components(tr_text, te_text, config)
        print("total preprocessing:", time.time() - start, "seconds\n")
        return self.components
            
    def get_train_test_embeddings(self, config, train_ratio=0.7):
        """Return the concatenated train-test feature matrices for the classificaiton task."""
        _ = self.prepare_train_test_components(config, train_ratio)
        X_train, X_test = self._generate_embeddings(config)
        return X_train, X_test
    
    def export(self, output_dir, config, train_ratio=0.7, delimiter=","):
        """Export features, labels and other meta data for a given train-test data split."""
        model_dir = os.path.join(output_dir, self.model_id)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        X_tr, X_te = self.get_train_test_embeddings(config, train_ratio)
        #save text feature matrix
        idx = config["dimension"]
        tr_text = csr_matrix(X_tr[:,:idx])
        te_text = csr_matrix(X_te[:,:idx])
        print("text", tr_text.shape, te_text.shape)
        save_npz(os.path.join(model_dir, "train_text"), tr_text)
        save_npz(os.path.join(model_dir, "test_text"), te_text)
        #save history feature matrix
        if config.get("user_history", False):
            tr_history = X_tr[:,idx:idx+4]
            te_history = X_te[:,idx:idx+4]
            np.savetxt(os.path.join(model_dir, "train_history.csv"), tr_history, delimiter=delimiter)
            np.savetxt(os.path.join(model_dir, "test_history.csv"), te_history, delimiter=delimiter)
            idx += 4
            print("history", tr_history.shape, te_history.shape)
        # save node embeddings
        if "user_ne" in config and X_tr.shape[1] > idx:
            tr_network = X_tr[:,idx:]
            te_network = X_te[:,idx:]
            np.savetxt(os.path.join(model_dir, "train_network.csv"), tr_network, delimiter=delimiter)
            np.savetxt(os.path.join(model_dir, "test_network.csv"), te_network, delimiter=delimiter)
            print("network", tr_network.shape, te_network.shape)
        #save labels
        np.savetxt(os.path.join(model_dir, "train_label.csv"), self.tr_label, delimiter=delimiter, fmt='%i')
        np.savetxt(os.path.join(model_dir, "test_label.csv"), self.te_label, delimiter=delimiter, fmt='%i')
        #save meta
        self.tr_meta[self._exported_meta_columns].to_csv(os.path.join(model_dir, "train_meta.csv"), index=False, sep=delimiter)
        self.te_meta[self._exported_meta_columns].to_csv(os.path.join(model_dir, "test_meta.csv"), index=False, sep=delimiter)
        print("Model was exported")
        return model_dir
