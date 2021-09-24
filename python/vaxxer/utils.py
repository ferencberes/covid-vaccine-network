import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

##############
# Data reading
##############

def original_data(thread_fp, nrows=None):
    """Read raw Twitter data"""
    print("Loading threads from: %s" % thread_fp)
    # reading in data
    thread_df = pd.read_csv(thread_fp, nrows=nrows)
    # converting id fields from string to int
    thread_df['id_str'] = thread_df['id_str'].astype(int)
    thread_df['usr_id_str'] = thread_df['usr_id_str'].astype(int)
    thread_df['conversation_id'] = thread_df['conversation_id'].astype(int)
    # filtering the dataframe to only contain english language tweets
    thread_df = thread_df[thread_df.lang == "en"]
    seed_df = thread_df[thread_df["is_conv_seed"]]
    return thread_df, seed_df

def labeled_data(label_fp):
    """Read (manually-labeled) vaxxer annotations from file"""
    print("Loading labels from: %s" % label_fp)
    # reading in labeled data
    label_df = pd.read_csv(label_fp, names=["timestamp", "id_str", "usr_id_str", "is_conv_seed", "label", "label_id"])
    label_df = label_df.drop_duplicates(subset="id_str", keep="last")
    label_df['id_str'] = label_df['id_str'].astype(int)
    return label_df

#####################
# Get the clean data
#####################

def temporal_train_test_split(labeled_df, train_ratio=0.6, time_col="epoch", verbose=True):
    """Temporal train-test split for labeled tweets"""
    labeled_seeds_df = labeled_df.sort_values(time_col, ascending=True).copy()
    size = len(labeled_seeds_df)
    train_size = int(size*train_ratio)
    test_size = size-train_size
    if verbose:
        print("Train size:", train_size)
        print("Test size:", test_size)
    train_df = labeled_seeds_df.head(train_size)
    test_df = labeled_seeds_df.tail(test_size)
    if verbose:
        print("Train label distribution:")
        print(train_df["label"].value_counts().sort_index() / len(train_df))
        print("Test label distribution:")
        print(test_df["label"].value_counts().sort_index() / len(test_df))
    return train_df, test_df

def clean_labeled_data(seed_df, drop_irrelevant=False, verbose=True):
    """Exclude low-frequency or invalid vaxxer categories"""
    # "Don't know" and "Pro-choice" are excluded!
    if drop_irrelevant:
        seed_df = seed_df[seed_df["label"] != "Irrelevant"]
    seed_df = seed_df[seed_df["label"] != "Don't know"]
    seed_df = seed_df[seed_df["label"] != "Pro-choice"]
    labeled_df = seed_df[~seed_df["label"].isnull()]
    if verbose:
        print("Number of labeled records after cleaning:", len(labeled_df))
    unlabeled_df = seed_df[seed_df["label"].isnull()]
    return labeled_df, unlabeled_df

class LabelIndexer():
    def __init__(self):
        self._options = ["Pro-vaxxer", "Irrelevant"]
        # "Don't know" and "Pro-choice" are excluded!
        self._other = "Vax-skeptic"
            
    @property
    def labels(self):
        labels = self._options + [self._other]
        return labels.copy()
            
    @property
    def num_categories(self):
        return len(self._options) + 1
            
    def label2idx(self, label):
        if label in self._options:
            return self._options.index(label)
        else:
            return len(self._options)
        
    def idx2label(self, index):
        if index < len(self._options):
            return self._options[index]
        else:
            return self._other
        
    def get_index_mapping(self):
        return {idx:self.idx2label(idx) for idx in range(self.num_categories)}

def text_and_label(df):
    """Encode text labels to numerical values"""
    indexer = LabelIndexer()
    texts = df["full_text"].values
    labels = [indexer.label2idx(label) for label in df.label.values]
    return texts, labels

def get_train_test_data(thread_fp, label_fp, train_ratio=0.6, meta_cols=["id_str","usr_id_str"], drop_irrelevant=False, visualize=True, verbose=True):
    """Load data with temporal train-test split"""
    # reading in data
    thread_df, seed_df = original_data(thread_fp)
    label_df = labeled_data(label_fp)
    if verbose:
        print("Number of labeled records:", len(label_df))
    # joining the seed dataframe with the labels by id_str
    seed_df = pd.merge(seed_df, label_df.drop(['usr_id_str', 'is_conv_seed'], axis=1), on="id_str", how="left")
    labeled_df, unlabeled_df = clean_labeled_data(seed_df, drop_irrelevant, verbose)
    if visualize:
        fig, ax = plt.subplots(1,1,figsize=(15,6))
        plt.title("Number of daily labeled tweets")
        labeled_df["date"].value_counts().sort_index().plot(ax=ax)
    train_df, test_df = temporal_train_test_split(labeled_df, train_ratio, verbose=verbose)
    tr_text, tr_label = text_and_label(train_df)
    te_text, te_label = text_and_label(test_df)
    tr_meta, te_meta = train_df[meta_cols], test_df[meta_cols]
    return tr_text, tr_label, tr_meta, te_text, te_label, te_meta, unlabeled_df

################
# Classification
################
    
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

def metrics2df(metrics_dict):
    records = []
    for model in metrics_dict:
        for part in metrics_dict[model]:
            for metric in metrics_dict[model][part]:
                score = metrics_dict[model][part][metric]
                records.append([model, part, metric, score])
    return pd.DataFrame(records, columns=["model","part","metric","score"])

def calculate_metrics(model, X, y, multiclass=False, show_confusion_matrix=False, verbose=True):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    if show_confusion_matrix:
        cm = confusion_matrix(y, y_pred)
        print(model)
        print(cm)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro' if multiclass else 'binary')
    if verbose:
        print("Accuracy:", acc)
        print("F-score (macro):", f1)
    auc = None
    if not multiclass:
        auc = roc_auc_score(y, y_proba[:, 1])
        if verbose:
            print("Roc Auc score:", auc)
    metrics = {
        "acc":acc,
        "f1":f1,
        "auc":auc,
    }
    return metrics, y_pred, y_proba

def evaluate_classifier(model, data, multiclass=False, show_confusion_matrix=False, verbose=True):
    X_train, X_test, y_train, y_test = data
    model = model.fit(X_train, y_train)
    results = {}
    if verbose:
        print(model)
        print("TRAIN:")
    results["train"] = calculate_metrics(model, X_train, y_train, multiclass, show_confusion_matrix, verbose)[0]
    if verbose:
        print("TEST:")
    results["test"] = calculate_metrics(model, X_test, y_test, multiclass, show_confusion_matrix, verbose)[0]
    if verbose:
        print()
    return results

def evaluate_classifiers(model_tuples, vectors, labels, vectors_test=None, labels_test=None, test_size=0.3, multiclass=False, show_confusion_matrix=False, verbose=True):
    names, classifiers = zip(*model_tuples)
    if vectors_test is None:
        train_test_data = train_test_split(vectors, labels, random_state=0, test_size=test_size)
    else:
        train_test_data = (vectors, vectors_test, labels, labels_test)
    results = {}
    for i in range(len(classifiers)):
        results[names[i]] = evaluate_classifier(classifiers[i], train_test_data, multiclass, show_confusion_matrix, verbose)
    metrics_df = metrics2df(results)
    metrics_df["dimension"] = vectors.shape[1]
    return metrics_df

from collections import deque
def dynamic_auc(df, window=7*86400):
    """Calculate AUC over time with the pre-defined time window."""
    q = deque([])
    last_date = None
    metrics = []
    for _, row in df.sort_values("epoch").iterrows():
        if last_date == None:
            last_date = row["date"]
        record = (row["epoch"], row["label"], row["proba"])
        if row["date"] != last_date:
            current_time = record[0]
            while current_time - q[0][0] > window:
                q.popleft()
            arr = np.array(q)
            if len(arr) > 1:
                auc = roc_auc_score(arr[:,1], arr[:,2])
                metrics.append((last_date, auc))
            last_date = row["date"]
        q.append(record)
    return pd.DataFrame(metrics, columns=["date","score"])

###############
# Visualization
###############

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_dynamic_auc(configs, predictions, badrate,  window=7*86400, markers=["circle","square","star","triangle-up"]):
    """Show AUC over time with the pre-defined time window."""
    parts = []
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for idx, key in enumerate(configs):
        if idx == 3:
            continue
        tmp_auc = dynamic_auc(predictions[idx], window)
        tmp_auc["experiment_id"] = key
        parts.append(tmp_auc)
        fig.add_trace(
        go.Scatter(
            x=tmp_auc["date"], 
            y=tmp_auc["score"], 
            name=key, 
            mode='lines+markers',
            marker_symbol=markers[idx]
        ),
        row=1, col=1, secondary_y=False)
    metrics_df = pd.concat(parts)
    dates = list(metrics_df["date"].unique())
    fig.add_trace(
    go.Scatter(x=dates, y=badrate[dates], name="Vax-skeptic rate", line=dict(dash="dot")),
        row=1, col=1, secondary_y=True)
    fig.update_layout(
        yaxis_title="AUC",
        font_size=18,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    fig.update_yaxes(title_text="Vax-skeptic rate", secondary_y=True)
    return fig

################
# Model stacking
################

def calculate_sequential_stats(df, id_col, user_col, value_cols=None, aggregations=["mean"]):
    """
    Aggregate user statistics sequentially. The input dataframe 'df' must be pre-ordered!
    """
    if value_cols == None:
        value_cols = list(df.columns)
        value_cols.remove(id_col)
        value_cols.remove(user_col)
    user_records = {}
    agg_records = {agg_type:[] for agg_type in aggregations}
    for idx in tqdm(range(len(df)), mininterval=60):
        row = df.iloc[idx]
        id_, user_, values_ = str(row[id_col]), str(row[user_col]), list(row[value_cols])
        #print(user_)
        if len(user_records.get(user_,[])) > 0:
            user_history = pd.DataFrame(user_records[user_], columns=value_cols)
            for agg_type in aggregations:
                agg_records[agg_type].append(list(user_history.agg(agg_type)))
        else:
            for agg_type in aggregations:
                agg_records[agg_type].append([None] * len(value_cols))
        if pd.isnull(values_).sum() == 0:
            if not user_ in user_records:
                user_records[user_] = []
            user_records[user_].append(values_)
    agg_df = df.copy()
    for agg_type in aggregations:
        tmp_df = pd.DataFrame(agg_records[agg_type], index=agg_df.index, columns=[agg_type+"_"+col for col in value_cols])
        agg_df = pd.concat([agg_df, tmp_df], axis=1)
    return agg_df

def append_meta_cols(X, meta_df, cols):
    df_with_ids = pd.concat([meta_df[cols], pd.DataFrame(X, index=meta_df.index)], axis=1)
    if df_with_ids.isnull().sum().max() > 0:
        raise RuntimeError("Missing values occured after concatenation!")
    return df_with_ids

def get_user_stats(df_with_ids, user_id="usr_id_str", record_id="id_str", agg_type="mean"):
    user_df = df_with_ids.drop(record_id, axis=1).groupby(user_id).agg(agg_type).reset_index()
    renamed_cols = ["%s_usr_%s" % (col, agg_type) if (col != user_id) and (col != record_id) else col for col in user_df.columns]
    user_df.columns = renamed_cols
    return user_df

def get_twitter_stats(meta_df, record_id="id_str", cols=['usr_followers_count', 'usr_friends_count', 'usr_favourites_count', 'usr_statuses_count']):
    keep_cols = [record_id] + cols
    tmp_df = meta_df[keep_cols].copy()
    for col in cols:
        tmp_df[col] = np.log(1+tmp_df[col]).astype("int")
    return tmp_df
    
def merge_features_parts(parts, key_col):
    merged_df = parts[0]
    for idx in range(1, len(parts)):
        common_cols = set(merged_df.columns).intersection(set(parts[idx]))
        if len(common_cols) != 1:
            raise RuntimeError("Number of common columns is %i" % len(common_cols))
        merged_df = merged_df.merge(parts[idx], on=key_col, how="left")
    return merged_df

def load_node_embeddings(file_path, user_id="usr_id_str", users=None):
    node_emd_df = pd.read_csv(file_path, header=None)
    node_emd_df.rename({0:user_id}, axis=1, inplace=True)
    print("Original size:", node_emd_df.shape)
    if users != None:
        node_emd_df = node_emd_df[node_emd_df[user_id].isin(users)]
        print("Filtered size:", node_emd_df.shape)
    renamed_cols = ["%s_ne" % col if col != user_id else col for col in node_emd_df.columns]
    node_emd_df.columns = renamed_cols
    return node_emd_df

def user_label_encoding(tr_meta, tr_labels, te_meta, te_labels, record_id="id_str", user_id="usr_id_str"):
    p1 = tr_meta.copy()
    p1["LABEL"] = tr_labels
    p2 = te_meta.copy()
    p2["LABEL"] = te_labels
    tmp_df = pd.concat([p1,p2], axis=0)
    null_counts = tmp_df.drop("LABEL", axis=1).isnull().sum()
    if null_counts.max() > 0:
        print(null_counts.sort_values())
        raise RuntimeError("There are missing values in the feature matrix!")
    agg_df = calculate_sequential_stats(tmp_df, record_id, user_id, value_cols=["LABEL"], aggregations=["mean","std","min","max"])
    #remove label column from the feature matrix!
    agg_df.drop("LABEL", axis=1, inplace=True)
    agg_df = agg_df.fillna(agg_df.mean())
    tr_cleaned = agg_df.head(len(tr_meta)).copy()
    te_cleaned = agg_df.tail(len(te_meta)).copy()
    return tr_cleaned, te_cleaned
