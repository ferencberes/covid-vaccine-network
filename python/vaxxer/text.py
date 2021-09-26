import re, unicodedata, contractions, emoji, inflect, nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np

### noise removal ###
def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_urls(text):
    """Remove urls from text"""
    return re.sub('(https:|http:|www\.)\S*', '', text)

def remove_hashtags(text):
    """Remove '#' from hashtags. The hashtag text is kept as it is an organic part of the tweets"""
    return re.sub('#+', '', text)

def remove_mentions(text):
    """Remove @-mentions from the text"""
    return re.sub('@\w+', '', text)

def replace_and_sign(text):
    """Replace '\&amp;' with 'and' in the text"""
    return re.sub('\&amp;', ' and ', text)

def replace_contractions(text):
    """Handle contractions in the text"""
    return contractions.fix(text)

def replace_emojis(text):
    """Replace emojis with the related string"""
    return emoji.demojize(text, delimiters=(" em_", " "))

def replace_and_extract_emojis(text):
    """Replace emojis with the related string then return emojis as well"""
    demojized_text = emoji.demojize(text, delimiters=(" em_", " "))
    encoded_emojis = re.findall(r'\b[em_]\w+', demojized_text)
    return demojized_text, encoded_emojis

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_word = re.sub(r'-| |,', '', new_word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    # also removes empty strings
    new_words = []
    for word in words:
        new_word = re.sub(r' |[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    # replaces non-ascii tokens with ""
    # example £ §
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

### full text preprocessing ###

def extract_emojis(text):
    demojized_text = emoji.demojize(text, delimiters=(" em_", " "))
    raw_emojis = []
    for emoji_dict in emoji.emoji_lis(text):
        raw_emojis.append(emoji_dict['emoji'])
    return raw_emojis

def denoise_text(text, emojis=True):
    """Denoise text by removing urls, hashtags, @-mentions and emojis."""
    text = remove_urls(text)
    text = remove_hashtags(text)
    text = remove_mentions(text)
    text = replace_and_sign(text)
    text = replace_contractions(text)
    if emojis:
        text = replace_emojis(text)
    return text

def normalize(words):
    """Normalize text by removing special characters, punctuations, stopwords and replacing numbers with text. The normalized text is further converted to lowercase."""
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = to_lowercase(words)
    words = remove_stopwords(words)
    return words

def tokenize_text(text):
    """Tokenize text"""
    token_list = word_tokenize(text)
    return token_list

def text_preprocessing(text: str):
    """Text preprocessing pipeline: denoising, tokenization, normalization."""
    text = denoise_text(text)
    word_list = tokenize_text(text)
    word_list = normalize(word_list)
    return word_list

### additional functionality ###
def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def stem_and_lemmatize(words):
    """Stem and lemmatize words"""
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas

def w2v_infer_vector(model, sentence):
    """Infer sentence embedding for gensim.Word2Vec model."""
    keys = [w for w in sentence if w in model.wv]
    if len(keys) > 0:
        return np.mean(model.wv[keys], axis=0)
    else:
        return np.zeros(model.vector_size)

### preprocessing for multiple sentences ###

def preprocessing_tfidf(sentences):
    """Sentence preprocessing pipeline for TF-IDF. Normal text is returned."""
    preprocessed_data = []
    for sentence in sentences:
        preprocessed_data.append(' '.join(text_preprocessing(sentence)))
    return preprocessed_data

def preprocessing_document_list(sentences):
    """Sentence preprocessing pipeline for 'gensim'. Tokenized text is returned."""
    tokenized_data = []
    for sentence in sentences:
        tokenized_data.append(text_preprocessing(sentence))
    return tokenized_data
