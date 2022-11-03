#!/bin/bash
components_dir=$1
parameters_path=$2
bert=$3
gpu_device=$4
result_prefix=$5
instances=$6
comet_key_path=$7

python run_classifier.py $components_dir --fix $bert --exclude "History" "Twitter" "Centrality" "Network" "TfIdf" --num_instances $instances --algo "GRID" --parameters $parameters_path --num_trials 500 --neural_model "lstm" --device $gpu_device --experiment_folder $result_prefix"only_text" --comet_file $comet_key_path

#python run_classifier.py $components_dir --fix $bert "History/norm:True" --exclude "Twitter" "Centrality" "Network" "TfIdf" --num_instances $instances --algo "GRID" --parameters $parameters_path --num_trials 500 --neural_model "lstm" --device $gpu_device --experiment_folder $result_prefix"text+history" --comet_file $comet_key_path

#python run_classifier.py $components_dir --fix $bert "Twitter/norm:True" "Centrality/norm:True" --exclude "History" "Network" "TfIdf" --num_instances $instances --algo "GRID" --parameters $parameters_path --num_trials 500 --neural_model "lstm" --device $gpu_device --experiment_folder $result_prefix"text+twitter+centrality" --comet_file $comet_key_path

#python run_classifier.py $components_dir --fix $bert "Network/norm:True_user_ne:Walklets_dim128_2021-09-05_12:03:25.csv" --exclude "History" "Twitter" "Centrality" "TfIdf" --num_instances $instances --algo "GRID" --parameters $parameters_path --num_trials 500 --neural_model "lstm" --device $gpu_device --experiment_folder $result_prefix"text+network" --comet_file $comet_key_path

#python run_classifier.py $components_dir --fix $bert "History/norm:True" "Network/norm:True_user_ne:Walklets_dim128_2021-09-05_12:03:25.csv" --exclude "Twitter" "Centrality" "TfIdf" --num_instances $instances --algo "GRID" --parameters $parameters_path --num_trials 1 --neural_model "lstm" --device $gpu_device --experiment_folder $result_prefix"text+history+network" --comet_file $comet_key_path

#python run_classifier.py $components_dir --fix $bert "Twitter/norm:True" "Centrality/norm:True" "History/norm:True" "Network/norm:True_user_ne:Walklets_dim128_2021-09-05_12:03:25.csv" --exclude "TfIdf" --num_instances $instances --algo "GRID" --parameters $parameters_path --num_trials 500 --neural_model "lstm" --device $gpu_device --experiment_folder $result_prefix"text+history+twitter+centrality+network" --comet_file $comet_key_path