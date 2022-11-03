import sys, os, time, optuna, joblib, torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime as dt
from optuna.samplers import TPESampler, RandomSampler, GridSampler
from sklearn.linear_model import LogisticRegression
from transformers import logging
logging.set_verbosity_error()

from .utils import *
from .features import *
from .bert_utils import *
from .neural_models import *
from .comet_utils import init_experiment

def log_fixed_params(parameters: list, exp):
    """Log fixed parameters for comet experiment"""
    for c_name, c_type, c_vals in parameters:
        if c_type == "fixed":
            exp.log_parameter(c_name, c_vals)
            
def get_bert_model(str_setting:str):
    """Extract Huggingface BERT model identifier from string folder name"""
    out = None
    for argument in str_setting.split("_"):
        if "model:" in argument:
            out = argument.replace("model:","").replace("@","/")
            break
    return out

class VaxxerClassifier():
    def __init__(self, feature_dir: str, model: str=None, device: str=None, comet_key: str=None, verbose: bool=True):
        self._set_experiment_info(feature_dir)
        self.verbose = verbose
        self.model = model
        self.device = device
        self.comet_key = comet_key
        self._num_components = 0
        self._component_types = {}
        self._clear_cache()
        
    def _set_experiment_info(self, feature_dir: str):
        if not os.path.exists(feature_dir) or len(os.listdir(feature_dir)) == 0:
            raise RuntimeError("You must generate feature components before model training!")
        else:
            if "/" == feature_dir[-1]:
                self.feature_dir = feature_dir[:-1]
            else:
                self.feature_dir = feature_dir
            self.experiment_id = self.feature_dir.split("/")[-1]
            splitted = self.experiment_id.split("_")
            self.class_label = splitted[0]
            self.drop_irrelevant = splitted[1].replace("di","")
            self.train_ratio = float(splitted[2].replace("tr",""))
            
        
    def _clear_cache(self):
        self.tr_meta = None
        self.te_meta = None
        self.tr_label = None
        self.te_label = None
        self.components = []
        self.optional_components = ["History","Twitter","Network"]
        
    def _load_labels_and_meta(self, folders: list):
        if "Labels" in folders:
            label_dir = os.path.join(self.feature_dir, "Labels")
            self.tr_label = np.loadtxt(os.path.join(label_dir, "train_label.txt"))
            test_label_fp = os.path.join(label_dir, "test_label.txt")
            if os.path.exists(test_label_fp):
                self.te_label = np.loadtxt(test_label_fp)
                print("Train-test label size:", self.tr_label.shape, self.te_label.shape)
            else:
                self.te_label = None
            self.tr_meta = pd.read_csv(os.path.join(label_dir, "train_meta.csv"))
            self.te_meta = pd.read_csv(os.path.join(label_dir, "test_meta.csv"))
            print("Train-test meta size:", self.tr_meta.shape, self.te_meta.shape)
            # labels must be omitted during feature component loading
            folders.remove("Labels")
        else:
            raise RuntimeError("'Labels' folder must be present to load labels for the classification task!")
        
    def set_components(self, fixed: list, excluded: list):
        folders = os.listdir(self.feature_dir)
        self._load_labels_and_meta(folders)
        print("Available component folders:", folders)
        component_types = {}
        num_components = 0
        for folder in folders:
            if not folder in excluded:
                dir_path = os.path.join(self.feature_dir, folder)
                subfolders = []
                for f in os.scandir(dir_path):
                    if f.is_dir() and not os.path.join(folder, f.name) in excluded:
                        subfolders.append(f.name)
                component_types[folder] = subfolders
                num_components += max(1,len(subfolders))
        fixed_types = []
        for component_path in fixed:
            splitted = component_path.split("/")
            fixed_types.append(splitted[0])
            if len(splitted) == 2 and splitted[0] in component_types:
                num_components = num_components - len(component_types[splitted[0]]) + 1
                component_types[splitted[0]] = [splitted[1]]
            elif not splitted[0] in component_types:
                raise RuntimeError("'%s' component type is both present in the excluded and fixed components. Remove it from the excluded components!" % splitted[0])
            elif len(splitted) > 2:
                raise ValueError("Too long component path were set: %s. Provide only a relative path from the root of the feature directory!" % component_path)
        self._fixed_comp_types = fixed_types
        self._num_components = num_components
        self._component_types = component_types
        if "Bert" in self._component_types:
            self._embedding_type = "bert"
        else:
            self._embedding_type = "tfidf"
        print("Selected components:")
        print(self.parameters)
        
    @property
    def parameters(self):
        params = []
        for comp_type, comp_settings in self._component_types.items():
            if comp_type in self.optional_components:
                #if comp_type == "Network":
                assert len(comp_settings) > 0
                values = comp_settings.copy()
                if not comp_type in self._fixed_comp_types:
                    values.append("None")
            else:
                values= comp_settings.copy()
            params.append((comp_type, "choice", values))
        return params
        
    def _load_general_component(self, component_dir: str):
        out = []
        for fname in ["train.csv", "test.csv"]:
            df = pd.read_csv(os.path.join(component_dir, fname))
            out.append(df.values)
        # add column names
        out.append(list(df.columns))
        return out
    
    def _load_bert_component(self, component_dir: str):
        out = []
        for fname in ["train_ids.txt", "train_masks.txt", "test_ids.txt", "test_masks.txt"]:
            arr = torch.Tensor(np.loadtxt(os.path.join(component_dir, fname))).long()
            out.append(arr)
        # no columns in this case
        out.append(["bert_%i" for i in range(out[0].shape[1])])
        return out

    def load_components(self):
        print("Number of selected components:", self._num_components)
        components = {}
        with tqdm(total=self._num_components) as pbar:
            for comp_type, subfolders in self._component_types.items():
                components[comp_type] = {}
                if len(subfolders) > 0:
                    for subfolder in subfolders:
                        comp_dir = os.path.join(self.feature_dir, comp_type, subfolder)
                        if comp_type == "Bert":
                            components[comp_type][subfolder] = self._load_bert_component(comp_dir)
                        else:
                            components[comp_type][subfolder] = self._load_general_component(comp_dir)
                        pbar.update(1)
                else:
                    raise RuntimeError("%s component type should have multiple options!" % comp_type)
        if "Bert" in components or "TfIdf" in components:
            self.components = components
        else:
            raise RuntimeError("Tweet text representation in missing! Generate text representation with 'TfIdfComponent' or 'BertComponent'!")

    def _generate_embeddings(self, config: dict):
        """Generate tweet representations from different feature components"""   
        tr_parts = []
        te_parts = []
        all_columns = []
        for key, value in config.items():
            if key == "Bert":
                # handle Bert component later
                continue
            if key in self.components:
                if value == "None":
                    continue
                else:
                    tr_arr, te_arr, cols = self.components[key][value]
                tr_parts.append(tr_arr)
                te_parts.append(te_arr)
                all_columns.append(cols)
        if len(tr_parts) > 0:
            X_train = np.concatenate(tr_parts, axis=1)
            X_test = np.concatenate(te_parts, axis=1)
            print(X_train.shape, X_test.shape)
        else:
            X_train, X_test = None, None
        if self._embedding_type == "bert":
            train_ids, train_mask, test_ids, test_mask, cols = self.components["Bert"][config["Bert"]]
            y_tr = torch.Tensor(self.tr_label).long()
            if self.te_label is None:
                # create an artificial test label tensor to eliminate error messages:
                arr = np.zeros(len(self.te_meta))
                arr[0] = 1.0
                y_te = torch.Tensor(arr).long()
            else:
                y_te = torch.Tensor(self.te_label).long()
            all_columns = cols + all_columns
            if len(tr_parts) > 0:
                z_tr = torch.Tensor(X_train)
                z_te = torch.Tensor(X_test)
                X_train = TensorDataset(train_ids, train_mask, z_tr, y_tr)
                X_test = TensorDataset(test_ids, test_mask, z_te, y_te)
            else:
                X_train = TensorDataset(train_ids, train_mask, y_tr)
                X_test = TensorDataset(test_ids, test_mask, y_te)
        return X_train, X_test, all_columns
           
    def _run_single_config(self, config: dict, return_probas=False, show_conf_mx=False, verbose=False):
        """Run single experiment based on the parameters passed in the 'config' dictionary"""
        X_train, X_test, columns = self._generate_embeddings(config)
        if self._embedding_type != "bert" and self.model is None: #shallow classification
            solver = config.get("solver", "newton-cg")
            clf = LogisticRegression(solver=solver)
            metrics_df, train_probas, test_probas = fit_and_evaluate_sklearn_model(clf, X_train, X_test, self.tr_label, self.te_label, return_probas, show_conf_mx, verbose)
        else: #neuralmodel classification
            batch_size, epochs, lr_rate, schedule_type = config.get("batches",32), config.get("epochs",20), config.get("lr_rate",0.001), config.get("scheduler", "cosine")
            mode, inter_dim = config.get("mode","inter"), config.get("inter_dim",256)
            dropout = config.get("dropout", 0.0)
            maintain_lstm = config.get("maintain_lstm", True)
            num_labels = self.tr_label.shape[1] if "Multiclass" == self.class_label else 2
            if self._embedding_type != "bert": #NeuralModel classification
                input_dim = X_train[0][0].size().numel()
                if self.model == "lstm":
                    lstm_dim, maintain_lstm = config["inter_dim"], maintain_lstm
                    clf = LSTMClassifier(num_labels, batch_size, epochs, lr_rate, schedule_type, 
                            input_dim, mode, inter_dim, dropout, use_lstm=True, lstm_dim=2*inter_dim,
                            maintain_lstm_state=maintain_lstm, device=self.device, show_confusion_matrix=show_conf_mx, verbose=verbose)
                else:
                    clf = TextClassifier(num_labels, batch_size, epochs, lr_rate, schedule_type, 
                            input_dim, mode, inter_dim, dropout, device=self.device, show_confusion_matrix=show_conf_mx, verbose=verbose)
            else: #bert
                bert_model = get_bert_model(config["Bert"])
                bert_dropout = config.get("bert_dropout", 0.0)
                train_bert = config.get("train_bert", False)
                if self.model == "lstm":
                    stack_dim = sum(x.size().numel() for x in X_train[0][2:-1])
                    clf = BertLSTMClassifier(num_labels, batch_size, epochs, lr_rate, schedule_type, 
                            stack_dim, mode, inter_dim, dropout, bert_model, bert_dropout, train_bert,
                            use_lstm=True, lstm_dim=2*inter_dim, maintain_lstm_state=maintain_lstm, device=self.device, show_confusion_matrix=show_conf_mx, verbose=verbose)
                else:
                    stack_dim = sum(x.size().numel() for x in X_train[0][2:-1])
                    clf = BertClassifier(num_labels, batch_size, epochs, lr_rate, schedule_type, stack_dim, mode,
                            inter_dim, dropout, bert_model, bert_dropout, train_bert, device=self.device, show_confusion_matrix=show_conf_mx, verbose=verbose)
            # evaluation is done through TensorDataset: no need to pass labels here
            metrics_df, train_probas, test_probas = clf.fit_eval_predict(X_train, X_test, return_probas)
        metrics_df["class"] = self.class_label
        config["model"] = clf.__class__.__name__
        #append parameter values to dataframe
        for key, value in config.items():
            metrics_df[key] = value
        return metrics_df, train_probas, test_probas
        
    def tune(self, external_parameters: list, search_alg: str, num_trials: int=5, metric: str="auc", direction: str="maximize", num_instances: int=1, comet_workspace: str=None, comet_project: str=None, export_dir=None):
        """Evaluate performance for multiple dimension parameter values"""
        parameters = self.parameters.copy()
        parameters += external_parameters
        print("Parameters:")
        print(parameters)

        if self.comet_key != None:
            exp = init_experiment(self.comet_key, comet_project, comet_workspace)
            exp.log_parameters({
                "multiclass":self.class_label,
                "train_ratio":self.train_ratio,
                "drop_irrelevant":self.drop_irrelevant,
                "trials":num_trials,
                "instances":num_instances,
                "metric":metric,
                "direction":direction,
                "search_alg":search_alg
            })
            log_fixed_params(parameters, exp)
        
        metric_df_parts = []
        def objective(trial):
            config = suggest_config(parameters, trial)
            instances = []
            for _ in range(num_instances):
                instance_df, _, _ = self._run_single_config(config)
                instances.append(instance_df)
            tmp_df = pd.concat(instances, axis=0)
            # aggregate performance for every configuration
            group_cols = list(tmp_df.drop("score", axis=1).columns)
            tmp_df = tmp_df.groupby(group_cols)["score"].agg(["mean","std"]).reset_index()
            tmp_df = tmp_df.sort_values(["part","metric"], ascending=True)
            metric_df_parts.append(tmp_df)
            print(tmp_df[["part","metric","mean","std"]])
            # optimize based on test performance
            test_df = tmp_df[tmp_df["part"]=="test"]
            metrics = dict(zip(test_df["metric"],test_df["mean"]))
            if self.comet_key != None:
                for name, value in metrics.items():
                    exp.log_metric(name, value, step=trial.number)
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
        if "model" not in best_config.keys():
            best_config["model"] = self.model
        print("Best config: ", best_config)
        
        if export_dir != None:
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
                os.chmod(export_dir, 0o777)
            self.export_probas(export_dir, parameters, best_config)
            study_fp = os.path.join(export_dir, "optuna_study.pkl")
            print("Study file:", study_fp)
            joblib.dump(study, study_fp)
            result_fp = os.path.join(export_dir, "metrics.csv")
            print("Output file:", result_fp)
            metrics_df.to_csv(result_fp, index=False)
            print("Study and metrics were exported!")
        
        if self.comet_key != None:
            exp.log_parameters(best_config)
            exp.log_metrics({
                "train_size":len(self.tr_meta),
                "test_size":len(self.te_meta),
                "score":study.best_value
            })
            exp.end()
            
        return best_config
    
    def export_probas(self, export_dir, parameters, best_config):
        augmented_best_config = augment_config(best_config, parameters)
        best_metrics, train_probas, test_probas = self._run_single_config(augmented_best_config, return_probas=True)
        print(best_metrics[["part","metric","score"]].sort_values(["part","metric"], ascending=True))
        print(train_probas.shape)
        tr_df = pd.DataFrame(train_probas)
        tr_df["label"] = self.tr_label
        tr_df = pd.concat([self.tr_meta, tr_df], axis=1)
        tr_df.to_csv(os.path.join(export_dir, "train_probas.csv"), index=False)
        print(test_probas.shape)
        te_df = pd.DataFrame(test_probas)
        te_df["label"] = self.te_label
        te_df = pd.concat([self.te_meta, te_df], axis=1)
        te_df.to_csv(os.path.join(export_dir, "test_probas.csv"), index=False)
        