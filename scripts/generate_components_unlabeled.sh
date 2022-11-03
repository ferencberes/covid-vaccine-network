#!/bin/bash
data_dir=$1
class_label=$2
train_ratio=1.0

# BERT components
for bert in "digitalepidemiologylab/covid-twitter-bert" "ans/vaccinating-covid-tweets"
do
python run_feature_generator.py $data_dir --output_folder "experiments/unlabeled_prediction" --class_label $class_label --train_ratio $train_ratio bert --model $bert
done

# additional feature components
for comp in "history" "network" "twitter" "centrality" 
do
    python run_feature_generator.py $data_dir --output_folder "experiments/unlabeled_prediction" --class_label $class_label --train_ratio $train_ratio --normalize $comp;
done

