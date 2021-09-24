import sys, os, time
import numpy as np
from tqdm import tqdm
import pandas

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec

from .utils import *
from .text import *

class Feature():
    """Abstract object for a general feature component"""
    def __init__(self, tr_meta, te_meta):
        self.tr_meta = tr_meta
        self.te_meta = te_meta
        
    def preprocess(self):
        """Preprocess data before feature generation"""
        start = time.time()
        self._preprocess()
        print("Preprocessing %s:" % self.__class__.__name__, time.time() - start, "seconds\n")
    
    def generate(self, configuration):
        """Generate features based on the passed configuration"""
        start = time.time()
        X_train, X_test, columns = self._generate(configuration)
        if X_train is None:
            print("No features were returned by %s" % self.__class__.__name__)
        else:
            print("Generating %s features:" % self.__class__.__name__, time.time() - start, "seconds\n")
            print(X_train.shape, X_test.shape)
        return X_train, X_test, columns
    
class TextFeature(Feature):
    """Abstract object for a general text based feature component"""
    def __init__(self, tr_meta, te_meta, tr_text, te_text):
        super(TextFeature, self).__init__(tr_meta, te_meta)
        self.tr_text = tr_text
        self.te_text = te_text
        
    def _preprocess(self):
        self.tr_preproc = self._preprocess_sentences(self.tr_text)
        self.te_preproc = self._preprocess_sentences(self.te_text)
        print(len(self.tr_preproc), len(self.te_preproc))
        
class TfIdfComponent(TextFeature):
    """Tf-Idf based feature component"""
    def __init__(self, tr_meta, te_meta, tr_text, te_text):
        super(TfIdfComponent, self).__init__(tr_meta, te_meta, tr_text, te_text)
        
    def _preprocess_sentences(self, sentences):
        return preprocessing_tfidf(sentences)
    
    def _generate(self, config):
        dim = config["dimension"]
        use_svd = config.get("use_svd",False)
        if use_svd:
            vectorizer = TfidfVectorizer(max_features=None)
        else:
            vectorizer = TfidfVectorizer(max_features=dim)
        X_train = vectorizer.fit_transform(self.tr_preproc).toarray()
        X_test = vectorizer.transform(self.te_preproc).toarray()
        if use_svd:
            print("SVD is used")
            combined_arr = np.concatenate((X_train, X_test), axis=0)
            A = csc_matrix(combined_arr, dtype=float)
            u, s, vt = svds(A, k=dim)
            train_size = len(X_train)
            X_train = u[:train_size,:]
            X_test = u[train_size:,:]
            columns = ["tfidf_%i" % i for i in range(X_train.shape[1])]
        else:
            columns = vectorizer.vocabulary_
        return X_train, X_test, columns
    
class GensimComponent(TextFeature):
    """Gensim (Word2Vec, Doc2Vec) based feature component"""
    def __init__(self, tr_meta, te_meta, tr_text, te_text, model_type, workers=2):
        super(GensimComponent, self).__init__(tr_meta, te_meta, tr_text, te_text)
        self.workers = workers
        if model_type in ["word2vec","doc2vec"]:
            self.model_type = model_type
        else:
            raise ValueError("Choose 'model_type' from 'word2vec' or 'doc2vec'!")
            
    def _preprocess_sentences(self, sentences):
        return preprocessing_document_list(sentences)
    
    def _generate(self, config):
        dim = config["dimension"]
        if self.model_type == "doc2vec":
            tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(self.tr_preproc)]
            model = Doc2Vec(tagged_data, vector_size=dim, window=config["window"], min_count=config["min_count"], epochs=config["epochs"], workers=self.workers)
            X_train = [list(model.infer_vector(tokenized_data)) for tokenized_data in self.tr_preproc]
            X_test = [list(model.infer_vector(tokenized_data)) for tokenized_data in self.te_preproc]
            X_train = np.array(X_train)
            X_test = np.array(X_test)
        else:#word2vec
            model = Word2Vec(self.tr_preproc, vector_size=dim, window=config["window"], min_count=config["min_count"], epochs=config["epochs"], workers=self.workers)
            X_train = [list(w2v_infer_vector(model, tokenized_data)) for tokenized_data in self.tr_preproc]
            X_test = [list(w2v_infer_vector(model, tokenized_data)) for tokenized_data in self.te_preproc]
            X_train = np.array(X_train)
            X_test = np.array(X_test)
        columns = ["%s_%i" % (self.model_type, i) for i in range(X_train.shape[1])]
        return X_train, X_test, columns

class UserTwitter(Feature):
    """Extract Twitter statistics for users"""
    def __init__(self, tr_meta, te_meta, columns=['usr_followers_count', 'usr_friends_count', 'usr_favourites_count', 'usr_statuses_count'], record_id="id_str", user_id="usr_id_str"):
        super(UserTwitter, self).__init__(tr_meta, te_meta)
        self.record_id = record_id
        self.user_id = user_id
        self.columns = columns
        
    def _preprocess(self):
        self.tr_preproc = get_twitter_stats(self.tr_meta, self.record_id, self.columns)
        self.te_preproc = get_twitter_stats(self.te_meta, self.record_id, self.columns)
        
    def _generate(self, config):
        if config.get("user_twitter", False):
            train_df = self.tr_preproc[self.columns].fillna(0.0)
            test_df = self.te_preproc[self.columns].fillna(0.0)
            return train_df.values, test_df.values, self.columns
        else:
            return None, None, None
    
class UserHistory(Feature):
    """Aggregate historical labels for users"""
    def __init__(self, tr_meta, te_meta, tr_label, te_label, record_id="id_str", user_id="usr_id_str"):
        super(UserHistory, self).__init__(tr_meta, te_meta)
        self.tr_label = tr_label
        self.te_label = te_label
        self.record_id = record_id
        self.user_id = user_id
        
    def _preprocess(self):
        self.tr_preproc, self.te_preproc = user_label_encoding(self.tr_meta, self.tr_label, self.te_meta, self.te_label, self.record_id, self.user_id)
        
    def _generate(self, config):
        if config.get("user_history", False):
            train_df = self.tr_preproc.drop([self.record_id, self.user_id], axis=1).fillna(0.0)
            test_df = self.te_preproc.drop([self.record_id, self.user_id], axis=1).fillna(0.0)
            columns = [col for col in train_df.columns if "LABEL" in col]
            return train_df[columns].values, test_df[columns].values, columns
        else:
            return None, None, None
        
class NodeEmbedding(Feature):
    """Node embedding for users"""
    def __init__(self, tr_meta, te_meta, node_emb_dir, file_names=None, record_id="id_str", user_id="usr_id_str"):
        super(NodeEmbedding, self).__init__(tr_meta, te_meta)
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
                    embeddings = load_node_embeddings(fp, user_id=self.user_id, users=users)
                    self.embeddings[fname] = embeddings
                    # merge with records
                    coords = list(embeddings.columns)
                    coords.remove(self.user_id)
                    train_df = self.tr_meta.merge(embeddings, on=self.user_id, how="left")[coords]
                    test_df = self.te_meta.merge(embeddings, on=self.user_id, how="left")[coords]
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
            return train_df.values, test_df.values, columns
        else:
            return None, None, None