import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from tqdm import tqdm
import json, datetime

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
    if 'conversation_id' in thread_df.columns:
        thread_df['conversation_id'] = thread_df['conversation_id'].astype(int)
    # filtering the dataframe to only contain english language tweets
    if 'lang' in thread_df.columns:
        thread_df = thread_df[thread_df["lang"] == "en"]
    if 'is_conv_seed' in thread_df.columns:
        seed_df = thread_df[thread_df["is_conv_seed"]]
    else:
        seed_df = None
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
        print("Label distribution:")
        print(labeled_seeds_df["label"].value_counts().sort_index())
        print(labeled_seeds_df["label"].value_counts().sort_index() / len(labeled_seeds_df))
    train_df = labeled_seeds_df.head(train_size)
    test_df = labeled_seeds_df.tail(test_size)
    if verbose:
        print("Train size:", train_size)
        print("Test size:", test_size)
        print("Train label distribution:")
        print(train_df["label"].value_counts().sort_index())
        print(train_df["label"].value_counts().sort_index() / len(train_df))
        print("Test label distribution:")
        print(test_df["label"].value_counts().sort_index())
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
    if "label" in df.columns:
        labels = [indexer.label2idx(label) for label in df.label.values]
    else:
        labels = None
    return texts, labels

def get_train_test_data(thread_fp, label_fp, train_ratio=0.6, meta_cols=["id_str","usr_id_str"], drop_irrelevant=False, visualize=True, verbose=True, tweet_ids=None):
    """Load data with temporal train-test split"""
    # reading in data
    thread_df, seed_df = original_data(thread_fp)
    label_df = labeled_data(label_fp)
    if tweet_ids != None:
        thread_df = thread_df[thread_df["id_str"].isin(tweet_ids)]
        label_df = label_df[label_df["id_str"].isin(tweet_ids)]
        print("After filtering for tweet ids:", len(thread_df), len(label_df))
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
    unlabeled_text, _ = text_and_label(unlabeled_df)
    tr_meta, te_meta, unlabeled_meta = train_df[meta_cols], test_df[meta_cols], unlabeled_df[meta_cols]
    return tr_text, tr_label, tr_meta, te_text, te_label, te_meta, unlabeled_text, unlabeled_meta

def binary_labeled_data(thread_fp, label_fp, graph_use=False):
    """Function merged from tmich branch - might be deprecated..."""
    # reading in data
    thread_df, seed_df = original_data(thread_fp)
    label_df = labeled_data(label_fp)
    # joining the seed dataframe with the labels by id_str
    seed_df = pd.merge(seed_df, label_df.drop(['usr_id_str', 'is_conv_seed'], axis=1), on="id_str", how="left")
    labeled_df, unlabeled_df = clean_labeled_data(seed_df, False)
    tweet_texts = labeled_df["full_text"].values
    tweet_labels, tweet_labels = text_and_label(labeled_df)
    if not graph_use:
        return tweet_texts, tweet_labels, unlabeled_df
    else:
        seed_df['binary_label'] = tweet_labels
        return thread_df, seed_df
    
################
# Classification
################

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, log_loss

def metrics2df(metrics_dict):
    """Convert metrics dictionary into dataframe"""
    records = []
    for part in metrics_dict:
        for metric in metrics_dict[part]:
            score = metrics_dict[part][metric]
            records.append([part, metric, score])
    return pd.DataFrame(records, columns=["part","metric","score"])

def calculate_metrics(truth, probas, show_confusion_matrix=False, verbose=False):
    """Calculate performance metrics for a given setting"""
    num_classes = probas.shape[1]
    preds = np.argmax(probas, axis=1)
    metrics = {
        'f1': f1_score(truth, preds, average='macro' if num_classes > 2 else 'binary'),
        'acc': accuracy_score(truth, preds),
        'loss': log_loss(truth, probas)
    }
    if num_classes > 2:
        metrics["auc"] == None
    else:
        if num_classes == 2:
            pred = probas[:,1]
        else:
            pred = probas[:,0]
        metrics["auc"] = roc_auc_score(truth, pred)
    if verbose:
        print(metrics)
    if show_confusion_matrix:
        print(confusion_matrix(truth, preds))
    return metrics

def fit_and_evaluate_sklearn_model(model, X_train, X_test, y_train, y_test=None, return_probas=False, show_confusion_matrix=False, verbose=False):
    """Fit and evaluate scikit-learn model for the train and test set"""
    model = model.fit(X_train, y_train)
    train_probas = model.predict_proba(X_train)
    test_probas = model.predict_proba(X_test)
    results = {}
    results["train"] = calculate_metrics(y_train, train_probas, show_confusion_matrix, verbose)
    if not y_test is None:
        results["test"] = calculate_metrics(y_test, test_probas, show_confusion_matrix, verbose)
    if return_probas:
        train_probas = clf.predict(X_train)
        test_probas = clf.predict(X_test)
    else:
        train_probas, test_probas = None, None
    return metrics2df(results), train_probas, test_probas

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
def dynamic_auc(df, window=7*86400, epoch_col="epoch", label_col="label", proba_col="proba"):
    """Calculate AUC over time with the pre-defined time window."""
    tmp_df = df.copy()
    tmp_df["date"] = pd.to_datetime(tmp_df[epoch_col], unit="s").apply(lambda x: x.strftime("%Y-%m-%d"))
    q = deque([])
    last_date = None
    metrics = []
    for _, row in tmp_df.sort_values(epoch_col).iterrows():
        if last_date == None:
            last_date = row["date"]
        record = (row[epoch_col], row[label_col], row[proba_col])
        if row["date"] != last_date:
            current_time = record[0]
            while current_time - q[0][0] > window:
                q.popleft()
            arr = np.array(q)
            if len(arr) > 1:
                auc = roc_auc_score(arr[:,1], arr[:,2])
                mean_label = np.mean(arr[:,1])
                metrics.append((last_date, auc, mean_label))
            last_date = row["date"]
        q.append(record)
    return pd.DataFrame(metrics, columns=["date","auc","badrate"])

###############
# Visualization
###############

from wordcloud import WordCloud
from collections import Counter
from .text import text_preprocessing

def extract_frequent_word_by_label(seed_fp, label_fp, tweet_ids=None, forbidden = ['vaccine','vaccines','vaccinated','get','people','covid','covid19','vaccination','would','could','coronavirus','cellulitis']):
    tweet_text, tweet_label, _, _, _, _, _, _ = get_train_test_data(seed_fp, label_fp, train_ratio=1.0, drop_irrelevant=False, visualize=False, verbose=True, tweet_ids=tweet_ids)
    all_data = []
    skeptic_data = []
    provax_data = []
    irrelevant_data = []
    for i, sentence in enumerate(tweet_text):
        preprocessed_text = text_preprocessing(sentence, stem=False, lemmatize=True)
        all_data.append(preprocessed_text)
        if tweet_label[i] > 1:
            skeptic_data.append(preprocessed_text)
        elif tweet_label[i] > 0:
            irrelevant_data.append(preprocessed_text)
        else:
            provax_data.append(preprocessed_text)
    print(len(provax_data), len(skeptic_data), len(irrelevant_data))
    frequencies = Counter(flatten(all_data))
    all_data_freq = word_percentage_list(all_data)
    print(frequencies.most_common(10))
    skeptic_data_freq = word_percentage_list(skeptic_data)
    provax_data_freq = word_percentage_list(provax_data)
    irrelevant_data_freq = word_percentage_list(irrelevant_data)
    skeptic_high_freq = filter_high_freq_words(skeptic_data_freq, all_data_freq, forbidden)
    provax_high_freq = filter_high_freq_words(provax_data_freq, all_data_freq, forbidden)
    irrelevant_high_freq = filter_high_freq_words(irrelevant_data_freq, all_data_freq, forbidden)
    return skeptic_high_freq, provax_high_freq, irrelevant_high_freq, skeptic_data, provax_data, irrelevant_data, all_data_freq

def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

def word_percentage_list(nested_wordlist):
    n = len(nested_wordlist)
    counter_dict = dict(Counter(flatten(nested_wordlist)))
    for key, value  in counter_dict.items():
        counter_dict[key] = value/n
    return counter_dict

def filter_high_freq_words(data_freq, all_data_freq, forbidden=[]):
    high_freq = {}
    for key, value in data_freq.items():
        # filtering out emojis
        if "em_" not in key and key not in forbidden:
            if value > all_data_freq.get(key, 0):
                high_freq[key] = np.log10(value)
    return high_freq

import matplotlib.colors as colors

def show_wordcloud(word_counter, title=None, colormap="gnuplot", fsize=(20,10)):
    w_cloud = WordCloud(max_words=50, background_color='white', colormap=colormap, margin=10, width=fsize[0]*30, height=fsize[1]*30, 
                    scale=3, random_state=1).generate_from_frequencies(word_counter)
    fig = plt.figure(figsize=fsize)
    plt.imshow(w_cloud)
    if title != None:
        plt.title(title, fontsize=20)
    plt.axis("off")
    return fig
    #plt.show()
    #return w_cloud
    

def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def show_node_embedding_performance(network_auc_df, width=800, height=400):
    visu_df = network_auc_df.sort_values("mean")
    text_auc = float(visu_df[(visu_df["Network"]=="None") & (visu_df["History"]=="None")]["mean"])
    text_history_auc = float(visu_df[(visu_df["Network"]=="None") & (visu_df["History"]=="norm:True")]["mean"])
    print(text_auc, text_history_auc)
    visu_df = visu_df[visu_df["Network"]!="None"]
    visu_df["History"] = visu_df["History"].replace({"None":"text + raw network", "norm:True":"text + user history + raw network"}) 
    fig = px.bar(visu_df, x="Network", y="mean", color="History", barmode='group', width=width, height=height)
    #fig.update_yaxes(range=[visu_df["mean"].min()-0.01, visu_df["mean"].max()+0.01])
    fig.update_yaxes(range=[0.7, 0.9])
    fig.update_layout(
        #title="Performance with respect to different node embeddings",
        yaxis_title="AUC",
        xaxis_title=None,#"Node embedding",
        legend_title="Modalities used:",
        font_size=18,
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(yanchor="top", y=1.10, xanchor="left", x=0.02, orientation="h")
    )
    fig.add_hline(y=text_auc, line_dash="dot",
        annotation_text="<b>text</b>",
        line_color="blue",
        line_width=4,
        annotation_position="top left",
    )
    fig.add_hline(y=text_history_auc, line_dash="dash",
        annotation_text="<b>text + user history</b>",
        line_color="red",
        line_width=4,
        annotation_position="top left",
    )
    return fig

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

def filter_for_vaccine_view(visu_df, seeds_df, view_label, id_subset=None):
    """Filter the prepared data for visualization based on vaccine view"""
    filtered_visu_df = visu_df[visu_df["label"]==view_label].copy()
    filtered_users = list(filtered_visu_df["usr_id_str"].unique())
    filtered_tweets = seeds_df[seeds_df["usr_id_str"].isin(filtered_users)].copy()
    if id_subset != None:
        filtered_tweets = filtered_tweets[filtered_tweets["id_str"].isin(id_subset)]
    return filtered_visu_df, filtered_tweets

def divide_pro_and_skeptic(visu_df, seeds_df, id_subset=None):
    """Divide the prepared data for visualization based on vaccine view"""
    skeptic_df, skeptic_tweets = filter_for_vaccine_view(visu_df, seeds_df, 1, id_subset)
    pro_df, pro_tweets = filter_for_vaccine_view(visu_df, seeds_df, 0, id_subset)
    return skeptic_df, skeptic_tweets, pro_df, pro_tweets

from sklearn.decomposition import PCA

def dim_reduction(df, id_col="usr_id_str", cols=range(128), dim=2):
    """Reduce node embedding dimensions for visualization"""
    if cols != None:
        X = df[cols].values
    else:
        X = df.drop(id_col, axis=1).values
    print(X.shape)
    pca = PCA(n_components=dim)
    XX = pca.fit_transform(X)
    user_coords = dict(zip(df[id_col], XX))
    print(len(user_coords))
    return user_coords, XX

def merge_user_information(pred_df, X, coords=None, show_hist=False):
    """Extract user vaccine view and merge it with node embeddings"""
    # calculate mean user vaccine view
    mean_user_labels = pred_df.groupby("usr_id_str")["label"].mean()
    if show_hist:
        mean_user_labels.hist()
    label_map = dict(mean_user_labels)
    pred_tmp_df = pred_df.copy()
    # fix user view if it is not conclusive
    pred_tmp_df["label"] = pred_tmp_df["usr_id_str"].apply(lambda x: round(label_map[x]))
    # merge with node embedding information
    X_selected = X if coords == None else X[:,coords]
    part_1 = pd.DataFrame(X_selected, index=pred_tmp_df.index)
    part_2 = pred_tmp_df[["usr_id_str","label"]]
    visu_df = pd.concat([part_1, part_2], axis=1)
    # drop duplicated records of the same user
    visu_df = visu_df.drop_duplicates(subset="usr_id_str")
    return visu_df

from vaxxer.text import preprocess_sentences
from collections import Counter

def clean_text(df, text_col="full_text", new_col="clean_text", stem=False, lemmatize=True):
    df[new_col] = preprocess_sentences(list(df[text_col]), stem, lemmatize)
    cnt = Counter(df[new_col].str.split().sum())
    return df, cnt

def prepare_data_for_topic_visualization(user_df, seed_df, tweet_ids, stem=False, lemmatize=True):
    skeptic_df, skeptic_tweets, pro_df, pro_tweets = divide_pro_and_skeptic(user_df, seed_df, tweet_ids)
    print("Cleaning text STARTED...")
    skeptic_tweets, skeptic_word_counter = clean_text(skeptic_tweets, stem=stem, lemmatize=lemmatize)
    pro_tweets, pro_word_counter = clean_text(pro_tweets, stem=stem, lemmatize=lemmatize)
    print("Cleaning text FINISHED")
    skeptic_word_df = extract_word_representations(skeptic_df, skeptic_tweets)
    pro_word_df = extract_word_representations(pro_df, pro_tweets)
    return skeptic_word_df, pro_word_df

def extract_word_representations(user_df, tweets_df, id_col="usr_id_str", text_col="clean_text", dim=2, min_count=5, user_coords=None, stat_col=None, time_col=None):
    """Calculate word representations by aggregating user embeddings that included the given word in their tweets"""
    if user_coords == None:
        user_coords, _ = dim_reduction(user_df, id_col, dim=dim)
    word_coords = []
    for _, row in tweets_df.iterrows():
        usr_id = int(row[id_col])
        if usr_id in user_coords:
            coords = user_coords[usr_id]
            text_splitted = row[text_col].split()
            for word in text_splitted:
                record = [word, coords[0], coords[1]]
                if stat_col != None:
                    record.append(row[stat_col])
                if time_col != None:
                    record.append(row[time_col])
                word_coords.append(record)
    columns = ["word","x","y"]
    if stat_col != None:
        columns.append(stat_col)
    if time_col != None:
        columns.append(time_col)
    aggregated_columns = columns.copy()
    aggregated_columns.remove("word")
    new_columns = ["word"]
    if time_col != None:
        aggregated_columns.remove(time_col)
        new_columns.append(time_col)
    key_columns = new_columns.copy()
    for col in aggregated_columns:
        new_columns += [col+"_cnt", col+"_mean", col+"_std"]
    word_df = pd.DataFrame(word_coords, columns=columns)
    word_df = word_df.groupby(key_columns)[aggregated_columns].agg(["count","mean","std"]).reset_index()
    word_df.columns = new_columns
    word_df = word_df[word_df["x_cnt"]>min_count].sort_values("x_cnt", ascending=False)
    word_df["log_count"] = np.log10(1.0+word_df["x_cnt"])
    print(word_df.shape)
    return word_df

def filter_words_for_visualization(topics, word_df):
    """Filter word representations for the visualization"""
    color_map = {}
    for color, topic_words in topics.items():
        for word in topic_words:
            color_map[word]=color
    word_subset = word_df[word_df["word"].isin(list(color_map.keys()))].copy()
    word_subset["color"] = word_subset["word"].apply(lambda x: color_map[x])
    return word_subset

################
# Model stacking
################

def agg2str(agg_type):
    agg_type_str = str(agg_type)
    if "<function" in agg_type_str:
        agg_type_str = agg_type_str.split()[1]
    return agg_type_str

def add_new_record_to_user(user, values, user_records):
    if pd.isnull(values).sum() == 0:
        if not user in user_records:
            user_records[user] = []
        user_records[user].append(values)

def calculate_sequential_stats(df, id_col, user_col, value_cols=None, aggregations=["mean"], historical=True):
    """Aggregate user statistics sequentially. The input dataframe 'df' must be pre-ordered!"""
    if value_cols == None:
        value_cols = list(df.columns)
        value_cols.remove(id_col)
        value_cols.remove(user_col)
    user_records = {}
    agg_records = {agg2str(agg_type):[] for agg_type in aggregations}
    for idx in tqdm(range(len(df)), mininterval=5):
        row = df.iloc[idx]
        id_, user_, values_ = str(row[id_col]), str(row[user_col]), list(row[value_cols])
        if not historical:
            # add new record in advance
            add_new_record_to_user(user_, values_, user_records)
        if len(user_records.get(user_,[])) > 0:
            user_history = pd.DataFrame(user_records[user_], columns=value_cols)
            for agg_type in aggregations:
                agg_records[agg2str(agg_type)].append(list(user_history.agg(agg_type)))
        else:
            for agg_type in aggregations:
                agg_records[agg2str(agg_type)].append([None] * len(value_cols))
        if historical:
            # add new records only after historical aggregations
            add_new_record_to_user(user_, values_, user_records)
    agg_df = df.copy()
    for agg_type in aggregations:
        agg_type_str = agg2str(agg_type)
        tmp_df = pd.DataFrame(agg_records[agg_type_str], index=agg_df.index, columns=[agg_type_str+"_"+col for col in value_cols])
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

def get_twitter_stats(meta_df, record_id="id_str", user_id="usr_id_str", cols=['usr_followers_count', 'usr_friends_count', 'usr_favourites_count', 'usr_statuses_count'], logscale=True, use_int=True):
    keep_cols = [record_id, user_id] + cols
    tmp_df = meta_df[keep_cols].copy()
    if logscale:
        for col in cols:
            tmp_df[col] = np.log(1+tmp_df[col])
            if use_int:
                tmp_df[col] = tmp_df[col].astype("int")
    return tmp_df

def get_centralities(df, record_id, user_id, centrality_df):
    active_users = list(df[user_id].unique())
    centrality_small = centrality_df[centrality_df[user_id].isin(active_users)].copy()
    df_with_centrality = df[[record_id, user_id]].merge(centrality_small, on=user_id, how="left")
    return df_with_centrality
    
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

##################
#Parameter utils
##################

def space2json(params: list):
    """Export VaxxerClassifier search space to JSON file"""
    out = {}
    for setting in params:
        skey, stype, svalues = setting
        out[skey] = {
            "type":stype,
            "values":svalues
        }
    return out

def json2space(json_path: str):
    """Load VaxxerClassifier search space from JSON file"""
    with open(json_path) as f:
        json_config = json.load(f)
    params = []
    for key, config in json_config.items():
        setting_tuple = (key, config["type"], config["values"])
        params.append(setting_tuple)
    return params

def suggest_config(parameters: list, trial):
    """Convert manually defined search spaces into Optuna supported parameters suggestions"""
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

def augment_config(config, parameters):
    """Augment config with fixed parameters"""
    config_tmp = config.copy()
    for c_name, c_type, c_vals in parameters:
        if c_type == "fixed":
            config_tmp[c_name] = c_vals
    return config_tmp

def extract_grid(parameters: list):
    """Extract parameters for optuna.GridSampler"""
    grid = {}
    for c_name, c_type, c_vals in parameters:
        if c_type == "choice":
            grid[c_name] = c_vals
        elif c_type == "fixed":
            grid[c_name] = [c_vals]
        else:
            raise ValueError("GridSearch can only use categorical search space!")
    return grid