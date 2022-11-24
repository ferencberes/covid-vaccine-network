#!/bin/bash
mkdir -p data
cd data
# download tweet IDs
wget http://info.ilab.sztaki.hu/~fberes/covid_vaccine_data/tweet_ids_2022-11-14.zip
unzip tweet_ids_2022-11-14.zip
pushd tweet_ids_2022-11-14
popd
# download prepared feature components
wget http://info.ilab.sztaki.hu/~fberes/covid_vaccine_data/public_data_2022-10-27.zip
unzip public_data_2022-10-27.zip
pushd public_data_2022-10-27
for folder in "Pro-vaxxer_results" "Vax-skeptic_results"
do
mkdir $folder;
pushd $folder;
wget http://info.ilab.sztaki.hu/~fberes/covid_vaccine_data/parameters.json
popd
done
popd
