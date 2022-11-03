#!/bin/bash
data_dir=$1
class_label=$2
region_id=$3
train_ratio=0.7

# BERT components
for bert in "digitalepidemiologylab/covid-twitter-bert" "ans/vaccinating-covid-tweets"
do
python run_feature_generator.py $data_dir --output_folder "experiments/"$region_id"_"$region_id"_components" --class_label $class_label --train_ratio $train_ratio --tweet_filter $data_dir"/seed_preprocessed/"$region_id"_tweets.csv" bert --model $bert
done

# additional feature components
for comp in "history" "network" "twitter" "centrality"
do
    python run_feature_generator.py $data_dir --output_folder "experiments/"$region_id"_"$region_id"_components" --class_label $class_label --train_ratio $train_ratio --tweet_filter $data_dir"/seed_preprocessed/"$region_id"_tweets.csv" --normalize $comp
done;