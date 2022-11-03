#!/bin/bash
data_dir=$1
class_label=$2
train_ratio=0.7

# # BERT models
for model in "ans/vaccinating-covid-tweets" "digitalepidemiologylab/covid-twitter-bert" "digitalepidemiologylab/covid-twitter-bert-v2" "vinai/bertweet-covid19-base-uncased" "vinai/bertweet-base" "vinai/bertweet-large" "prajjwal1/bert-tiny" "prajjwal1/bert-mini" "prajjwal1/bert-small" "prajjwal1/bert-medium" "bert-base-uncased" "bert-large-uncased"
do
    python run_feature_generator.py $data_dir --output_folder "experiments/bert_components" --class_label $class_label --train_ratio $train_ratio bert --model $model
done