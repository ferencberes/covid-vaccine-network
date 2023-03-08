import sys, os, time, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from .utils import (
    LabelIndexer,
    get_train_test_data,
    user_label_encoding,
    get_twitter_stats,
    get_centralities,
    load_node_embeddings,
)
from .text import preprocess_sentences
from .bert_utils import preprocessing_bert


def dict2str(cfg: dict, normalize=False, prefix: str = None):
    out = "norm:%s_" % normalize
    if prefix != None:
        out = prefix + "_" + out
    for key, val in sorted(cfg.items(), key=lambda x: x[0]):
        out += str(key) + ":" + str(val) + "_"
    return out[:-1].replace("/", "@")


class Feature:
    """Abstract object for a general feature component"""

    def __init__(self, tr_meta, te_meta, normalize=False):
        self.tr_meta = tr_meta
        self.te_meta = te_meta
        self.normalize = normalize
        self.preprocessed = False

    def config(self, cfg: dict):
        return cfg.copy()

    def preprocess(self):
        """Preprocess data before feature generation"""
        start = time.time()
        self._preprocess()
        print(
            "Preprocessing %s:" % self.__class__.__name__,
            time.time() - start,
            "seconds\n",
        )
        self.preprocessed = True

    def generate(self, configuration):
        """Generate features based on the passed configuration"""
        if self.preprocessed:
            start = time.time()
            X_train, X_test, columns, setting_str = self._generate(configuration)
            if X_train is None:
                print("No features were returned by %s" % self.__class__.__name__)
            else:
                print(
                    "Generating %s features:" % self.__class__.__name__,
                    time.time() - start,
                    "seconds\n",
                )
                if isinstance(X_train, np.ndarray):
                    print(X_train.shape, X_test.shape)
                    if self.normalize:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                        print("Features were normalized!")
                else:
                    assert self.__class__.__name__ == "BertComponent"
            return X_train, X_test, columns, setting_str
        else:
            raise RuntimeError(
                "The preprocess() function was not executed! Please execute it before generating feature components!"
            )


class TextFeature(Feature):
    """Abstract object for a general text based feature component"""

    def __init__(
        self,
        tr_meta,
        te_meta,
        tr_text,
        te_text,
        stem=False,
        lemmatize=False,
        normalize=False,
    ):
        super(TextFeature, self).__init__(tr_meta, te_meta, normalize)
        self.tr_text = tr_text
        self.te_text = te_text
        self.stem = stem
        self.lemmatize = lemmatize
        print("Stemming:", self.stem, "Lemmatization:", self.lemmatize)

    def _preprocess(self):
        self.tr_preproc = self._preprocess_sentences(self.tr_text)
        self.te_preproc = self._preprocess_sentences(self.te_text)
        print(len(self.tr_preproc), len(self.te_preproc))

    def config(self, cfg: dict):
        tmp = super(TextFeature, self).config(cfg)
        tmp["stem"] = self.stem
        tmp["lemmatize"] = self.lemmatize
        return tmp


class TfIdfComponent(TextFeature):
    """Tf-Idf based feature component"""

    def __init__(
        self,
        tr_meta,
        te_meta,
        tr_text,
        te_text,
        stem=False,
        lemmatize=False,
        only_emoji=False,
        normalize=False,
    ):
        super(TfIdfComponent, self).__init__(
            tr_meta, te_meta, tr_text, te_text, stem, lemmatize, normalize
        )
        self.only_emoji = only_emoji

    def _preprocess_sentences(self, sentences):
        return preprocess_sentences(
            sentences,
            self.stem,
            self.lemmatize,
            only_emoji=self.only_emoji,
            lowercase=True,
            stopwords=True,
            tokenized_output=False,
        )

    def config(self, cfg: dict):
        tmp = super(TfIdfComponent, self).config(cfg)
        if self.only_emoji:
            del tmp["stem"]
            del tmp["lemmatize"]
            tmp["only_emoji"] = self.only_emoji
        return tmp

    def _generate(self, config):
        cfg = self.config(config)
        dim = cfg["dimension"]
        vectorizer = TfidfVectorizer(max_features=dim)
        X_train = vectorizer.fit_transform(self.tr_preproc).toarray()
        X_test = vectorizer.transform(self.te_preproc).toarray()
        columns = vectorizer.vocabulary_
        return X_train, X_test, columns, dict2str(cfg, self.normalize, "TfIdf")


class BertComponent(TextFeature):
    """BERT classifier based feature component"""

    def __init__(
        self,
        tr_meta,
        te_meta,
        tr_text,
        te_text,
        bert_model="ans/vaccinating-covid-tweets",
        max_tensor_len=120,
        stem=False,
        lemmatize=False,
        normalize=False,
    ):
        if normalize:
            raise ValueError("BertComponent cannot be normalized!")
        super(BertComponent, self).__init__(
            tr_meta, te_meta, tr_text, te_text, stem, lemmatize, normalize
        )
        self.bert_model = bert_model
        self.max_tensor_len = max_tensor_len

    def _preprocess_sentences(self, sentences):
        return preprocessing_bert(
            sentences, self.bert_model, self.max_tensor_len, self.stem, self.lemmatize
        )

    def config(self, cfg: dict):
        tmp = super(BertComponent, self).config(cfg)
        tmp["model"] = self.bert_model
        tmp["mtlen"] = self.max_tensor_len
        return tmp

    def _generate(self, config):
        cfg = self.config(config)
        input_ids, masks = self.tr_preproc
        print(type(input_ids), input_ids.shape)
        print(type(masks), masks.shape)
        te_input_ids, te_masks = self.te_preproc
        columns = ["bert_%i" % i for i in range(input_ids.size(1))]
        return (
            (input_ids, masks),
            (te_input_ids, te_masks),
            columns,
            dict2str(cfg, self.normalize, "Bert"),
        )


class UserTwitter(Feature):
    """Extract Twitter statistics for users"""

    def __init__(
        self,
        tr_meta,
        te_meta,
        columns=[
            "usr_followers_count",
            "usr_friends_count",
            "usr_favourites_count",
            "usr_statuses_count",
        ],
        record_id="id_str",
        user_id="usr_id_str",
        normalize=False,
    ):
        super(UserTwitter, self).__init__(tr_meta, te_meta, normalize)
        self.record_id = record_id
        self.user_id = user_id
        self.columns = columns

    def _preprocess(self):
        tr_preproc = get_twitter_stats(
            self.tr_meta,
            self.record_id,
            self.user_id,
            self.columns,
            logscale=True,
            use_int=False,
        )
        te_preproc = get_twitter_stats(
            self.te_meta,
            self.record_id,
            self.user_id,
            self.columns,
            logscale=True,
            use_int=False,
        )
        tmp_df = pd.concat([self.tr_meta, self.te_meta], axis=0)
        null_counts = tmp_df.isnull().sum()
        if null_counts.max() > 0:
            print(null_counts.sort_values())
            raise RuntimeError("There are missing values in the feature matrix!")

        def relative_change(x):
            return (1.0 + np.ptp(x)) / (1.0 + np.min(x))

        agg_df = (
            tmp_df.groupby(self.user_id)[self.columns]
            .agg([relative_change, "nunique"])
            .reset_index()
        )
        tr_preproc = tr_preproc.merge(agg_df, on=self.user_id, how="left")
        te_preproc = te_preproc.merge(agg_df, on=self.user_id, how="left")
        self.columns = list(
            tr_preproc.drop([self.record_id, self.user_id], axis=1).columns
        )
        self.tr_preproc = tr_preproc.fillna(tr_preproc.mean())
        self.te_preproc = te_preproc.fillna(te_preproc.mean())

    def _generate(self, config):
        if config.get("user_twitter", False):
            train_df = self.tr_preproc[self.columns].fillna(0.0)
            test_df = self.te_preproc[self.columns].fillna(0.0)
            return (
                train_df.values,
                test_df.values,
                self.columns,
                dict2str({}, self.normalize, "Twitter"),
            )
        else:
            return None, None, None, None


class UserHistory(Feature):
    """Aggregate historical labels for users"""

    def __init__(
        self,
        tr_meta,
        te_meta,
        tr_label,
        te_label,
        record_id="id_str",
        user_id="usr_id_str",
        normalize=False,
    ):
        super(UserHistory, self).__init__(tr_meta, te_meta, normalize)
        self.tr_label = tr_label
        self.te_label = te_label
        self.record_id = record_id
        self.user_id = user_id

    def _preprocess(self):
        self.tr_preproc, self.te_preproc = user_label_encoding(
            self.tr_meta,
            self.tr_label,
            self.te_meta,
            self.te_label,
            self.record_id,
            self.user_id,
        )

    def _generate(self, config):
        if config.get("user_history", False):
            train_df = self.tr_preproc.drop(
                [self.record_id, self.user_id], axis=1
            ).fillna(0.0)
            test_df = self.te_preproc.drop(
                [self.record_id, self.user_id], axis=1
            ).fillna(0.0)
            columns = [col for col in train_df.columns if "LABEL" in col]
            return (
                train_df[columns].values,
                test_df[columns].values,
                columns,
                dict2str({}, self.normalize, "History"),
            )
        else:
            return None, None, None, None


class UserCentrality(Feature):
    """Network centrality values for users"""

    def __init__(
        self,
        tr_meta,
        te_meta,
        centrality_file,
        record_id="id_str",
        user_id="usr_id_str",
        normalize=False,
    ):
        super(UserCentrality, self).__init__(tr_meta, te_meta, normalize)
        print(centrality_file)
        self.user_id = user_id
        self.record_id = record_id
        self.centrality_df = pd.read_csv(centrality_file)
        self.columns = list(self.centrality_df.columns)
        self.columns.remove(self.user_id)
        self.tr_preproc = {}
        self.te_preproc = {}

    def _preprocess(self):
        self.tr_preproc = get_centralities(
            self.tr_meta, self.record_id, self.user_id, self.centrality_df
        )
        self.te_preproc = get_centralities(
            self.te_meta, self.record_id, self.user_id, self.centrality_df
        )

    def _generate(self, config):
        if config.get("user_centrality", False):
            train_df = self.tr_preproc[self.columns].fillna(0.0)
            test_df = self.te_preproc[self.columns].fillna(0.0)
            return (
                train_df.values,
                test_df.values,
                self.columns,
                dict2str({}, self.normalize, "Centrality"),
            )
        else:
            return None, None, None, None


class NodeEmbedding(Feature):
    """Node embedding for users"""

    def __init__(
        self,
        tr_meta,
        te_meta,
        node_emb_dir,
        file_names=None,
        record_id="id_str",
        user_id="usr_id_str",
        normalize=False,
    ):
        super(NodeEmbedding, self).__init__(tr_meta, te_meta, normalize)
        self.record_id = record_id
        self.user_id = user_id
        self.node_emb_dir = node_emb_dir
        if file_names == None:
            self.file_names = os.listdir(self.node_emb_dir)
        else:
            self.file_names = file_names
        self.embeddings = {}
        self.tr_preproc = {}
        self.te_preproc = {}
        print(self.file_names)

    def _preprocess(self):
        users = set(self.tr_meta[self.user_id]).union(set(self.te_meta[self.user_id]))
        print(self.file_names)
        for fname in tqdm(self.file_names):
            fp = os.path.join(self.node_emb_dir, fname)
            if fname != "None" and not os.path.isdir(fp):
                try:
                    # load embeddings
                    embeddings = load_node_embeddings(
                        fp, user_id=self.user_id, users=users
                    )
                    self.embeddings[fname] = embeddings
                    # merge with records
                    coords = list(embeddings.columns)
                    coords.remove(self.user_id)
                    train_df = self.tr_meta.merge(
                        embeddings, on=self.user_id, how="left"
                    )[coords]
                    test_df = self.te_meta.merge(
                        embeddings, on=self.user_id, how="left"
                    )[coords]
                    train_df = train_df.fillna(0.0)
                    test_df = test_df.fillna(0.0)
                    self.tr_preproc[fname] = train_df
                    self.te_preproc[fname] = test_df
                except pandas.errors.ParserError:
                    print("Parsing error reading file: %s" % fp)
                except:
                    raise

    def _generate(self, config):
        fname = config.get("user_ne", "None")
        if fname in self.tr_preproc:
            train_df = self.tr_preproc[fname]
            test_df = self.te_preproc[fname]
            columns = ["ne_%i" % i for i in range(train_df.shape[1])]
            cfg = self.config(config)
            return (
                train_df.values,
                test_df.values,
                columns,
                dict2str(cfg, self.normalize, "Network"),
            )
        else:
            return None, None, None, None
