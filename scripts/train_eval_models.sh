#!/bin/bash
components_dir=$1
class_label=$2

if [ -z "$3" ]
then
    echo "Default number of instances is 1!"
    instances=1
else
    instances=$3
fi

if [ -z "$4" ]
then
    echo "Default gpu:0 is used!"
    gpu_device="cuda:0"
else
    gpu_device=$4
fi

if [ -z "$5" ]
then
    echo "No comet key file path was provided!"
    comet_key_path="None"
else
    comet_key_path=$5
fi

result_prefix=$components_dir$class_label"_results/"
parameters_file=$result_prefix"parameters.json"

# run BERT experiments
#python run_classifier.py $components_dir"bert_components/"$class_label"_diFalse_tr0.70/" --num_instances $instances --algo "GRID" --parameters $parameters_file --num_trials 500 --neural_model "lstm" --device $gpu_device --experiment_folder "bert_results" --comet_file $comet_key_path

if [ $class_label = "Vax-skeptic" ]
then
    bert="Bert/norm:False_lemmatize:False_model:ans@vaccinating-covid-tweets_mtlen:120_stem:False"
else
    bert="Bert/norm:False_lemmatize:False_model:digitalepidemiologylab@covid-twitter-bert_mtlen:120_stem:False"
fi

echo $bert

# run node embedding experiments
#python run_classifier.py $components_dir"network_components/"$class_label"_diFalse_tr0.70/" --fix $bert --num_instances $instances --algo "GRID" --parameters $parameters_file --num_trials 500 --neural_model "lstm" --device $gpu_device --experiment_folder "network_results" --comet_file $comet_key_path

# run modality experiments
bash train_eval_modalities.sh $components_dir"modality_components/"$class_label"_diFalse_tr0.70/" $parameters_file $bert $gpu_device $result_prefix $instances $comet_key_path

# run cross-region evaluation
#bash train_eval_modalities.sh $components_dir"eu_eu_components/"$class_label"_diFalse_tr0.70/" $parameters_file $bert $gpu_device $result_prefix"eu_eu_" $instances $comet_key_path
#bash train_eval_modalities.sh $components_dir"us_us_components/"$class_label"_diFalse_tr0.70/" $parameters_file $bert $gpu_device $result_prefix"us_us_" $instances $comet_key_path
#bash train_eval_modalities.sh $components_dir"eu_us_components/"$class_label"_diFalse_tr0.70/" $parameters_file $bert $gpu_device $result_prefix"eu_us_" $instances $comet_key_path
#bash train_eval_modalities.sh $components_dir"us_eu_components/"$class_label"_diFalse_tr0.70/" $parameters_file $bert $gpu_device $result_prefix"us_eu_" $instances $comet_key_path

# make prediction for unlabeled data
#python run_classifier.py $components_dir"unlabeled_prediction/"$class_label"_diFalse_tr1.00/" --fix $bert "History/norm:True" "Network/norm:True_user_ne:Walklets_dim128_2021-09-05_12:03:25.csv" --exclude "Twitter" "Centrality" --num_instances 1 --algo "GRID" --parameters $parameters_file --num_trials 1 --neural_model "lstm" --device $gpu_device --experiment_folder $result_prefix"unlabeled_prediction" --comet_file $comet_key_path