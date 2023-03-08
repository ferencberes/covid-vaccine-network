import sys, os, time
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import random_split
from transformers import BertTokenizer, AutoTokenizer

from .text import preprocess_sentences

BERTS = dict(
    [
        ("prajjwal1/bert-tiny", (2, 128)),
        ("prajjwal1/bert-mini", (4, 256)),
        ("prajjwal1/bert-small", (4, 512)),
        ("prajjwal1/bert-medium", (8, 512)),
        ("bert-base-uncased", (12, 768)),
        ("bert-large-uncased", (24, 1024)),
        ("vinai/bertweet-base", (None, None)),
        ("vinai/bertweet-large", (None, None)),
        ("vinai/bertweet-covid19-base-uncased", (None, None)),
        ("digitalepidemiologylab/covid-twitter-bert", (None, None)),
        ("ans/vaccinating-covid-tweets", (None, None)),
    ]
)


def preprocessing_bert(
    sentences,
    bert_model="bert-base-uncased",
    max_tensor_len=120,
    stem=False,
    lemmatize=False,
):
    preprocessed_text = preprocess_sentences(
        sentences,
        stem,
        lemmatize,
        lowercase=True,
        stopwords=True,
        tokenized_output=False,
    )
    if bert_model in [None, "None"]:
        raise ValueError(
            "Invalid BERT model was passed! Choose from:", list(BERTS.keys())
        )
    elif "vinai/bertweet" in bert_model:
        tokenizer = AutoTokenizer.from_pretrained(
            bert_model, normalization=True, use_fast=False, do_lower_case=True
        )
    elif (
        "digitalepidemiologylab" in bert_model
        or bert_model == "ans/vaccinating-covid-tweets"
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            bert_model, use_fast=False, do_lower_case=True
        )
    else:  # small berts + original berts
        tokenizer = BertTokenizer.from_pretrained(
            bert_model, use_fast=False, do_lower_case=True
        )
    if type(preprocessed_text) != list:
        preprocessed_text = (
            preprocessed_text.tolist()
        )  # tokenizer expects a list(str) not numpyarray(str)
    encoded_dict = tokenizer(
        preprocessed_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_tensor_len,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded_dict["input_ids"]
    attention_masks = encoded_dict["attention_mask"]
    return input_ids, attention_masks
