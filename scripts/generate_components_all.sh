#!/bin/bash
data_dir=$1
class_label=$2

bash generate_components_modalities.sh $data_dir $class_label
bash generate_components_bert.sh $data_dir $class_label
bash generate_components_network.sh $data_dir $class_label
bash generate_components_regions.sh $data_dir $class_label "us"
bash generate_components_regions.sh $data_dir $class_label "eu"
bash generate_components_unlabeled.sh $data_dir $class_label