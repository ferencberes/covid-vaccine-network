#!/bin/bash
mkdir -p data
cd data
wget http://info.ilab.sztaki.hu/~fberes/covid_vaccine_data/public_data_2022-10-27.zip
unzip public_data_2022-10-27.zip
cd public_data_2022-10-27
for folder in "Pro-vaxxer_results" "Vax-skeptic_results"
do
mkdir $folder;
pushd $folder;
wget http://info.ilab.sztaki.hu/~fberes/covid_vaccine_data/parameters.json
popd
done