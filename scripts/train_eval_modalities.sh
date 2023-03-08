#!/bin/bash
components_dir=$1
parameters_path=$2
bert=$3
gpu_device=$4
result_prefix=$5
instances=$6
nn_model=$7
comet_key_path=$8
search_algo="GRID"
#search_algo="TPE"

python run_classifier.py $components_dir --fix $bert --exclude "History" "Twitter" "Centrality" "Network" --num_instances $instances --algo $search_algo --parameters $parameters_path --num_trials 500 --neural_model $nn_model --device $gpu_device --experiment_folder $result_prefix"only_text" --comet_file $comet_key_path --random_sample

python run_classifier.py $components_dir --fix $bert "History/norm:True" --exclude "Twitter" "Centrality" "Network" --num_instances $instances --algo $search_algo --parameters $parameters_path --num_trials 500 --neural_model $nn_model --device $gpu_device --experiment_folder $result_prefix"text+history" --comet_file $comet_key_path --random_sample

python run_classifier.py $components_dir --fix $bert "Twitter/norm:True" "Centrality/norm:True" --exclude "History" "Network" --num_instances $instances --algo $search_algo --parameters $parameters_path --num_trials 500 --neural_model $nn_model --device $gpu_device --experiment_folder $result_prefix"text+twitter+centrality" --comet_file $comet_key_path --random_sample

python run_classifier.py $components_dir --fix $bert "Network/norm:True_user_ne:Walklets_dim128_2021-09-05_12:03:25.csv" --exclude "History" "Twitter" "Centrality" --num_instances $instances --algo $search_algo --parameters $parameters_path --num_trials 500 --neural_model $nn_model --device $gpu_device --experiment_folder $result_prefix"text+network" --comet_file $comet_key_path --random_sample

python run_classifier.py $components_dir --fix $bert "Network/norm:True_user_ne:Walklets_dim128_2021-09-05_12:03:25.csv" "History/norm:True" --exclude "Twitter" "Centrality" --num_instances $instances --algo $search_algo --parameters $parameters_path --num_trials 500 --neural_model $nn_model --device $gpu_device --experiment_folder $result_prefix"text+history+network" --comet_file $comet_key_path
--random_sample

python run_classifier.py $components_dir --fix $bert "Twitter/norm:True" "Centrality/norm:True" "History/norm:True" "Network/norm:True_user_ne:Walklets_dim128_2021-09-05_12:03:25.csv" --num_instances $instances --algo $search_algo --parameters $parameters_path --num_trials 500 --neural_model $nn_model --device $gpu_device --experiment_folder $result_prefix"text+history+twitter+centrality+network" --comet_file $comet_key_path
 --random_sample