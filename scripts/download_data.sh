#!/bin/bash
mkdir -p data
cd data
#download train-test representations
wget http://info.ilab.sztaki.hu/~fberes/covid_vaccine_data/covid_vaxxer_representations_2021-09-24.zip
unzip covid_vaxxer_representations_2021-09-24.zip
#download raw Twitter data identifiers
#TODO
