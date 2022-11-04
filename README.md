# Vaccine skepticism detection by network embedding

[![Build](https://github.com/ferencberes/covid-vaccine-network/actions/workflows/main.yml/badge.svg)](https://github.com/ferencberes/covid-vaccine-network/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/ferencberes/covid-vaccine-network/branch/main/graph/badge.svg?token=B2D3JHO2K3)](https://codecov.io/gh/ferencberes/covid-vaccine-network)

[Ferenc Béres](https://github.com/ferencberes), [Rita Csoma](https://github.com/csomarita), [Tamás Vilmos Michaletzky](https://github.com/tmichaletzky), and [András A. Benczúr](https://mi.nemzetilabor.hu/people/andras-benczur)

In this work, we intended to develop techniques that are able to efficiently differentiate between pro-vaxxer and vax-skeptic Twitter content related to COVID19 vaccines. After multiple data preprocessing steps, we evaluated Tweet content and user interaction network classification by combining text classifiers with several node embedding and community detection techniques. 

Our work was published at the 10th International Conference on Complex Networks and Their Applications.

# Requirements

- UNIX environment
- Create a new conda environment with the related Python dependencies using the provided [YAML file](env.yml). 
- Activate the new conda environment **(covid\_vax\_network)**

```bash
conda env create -f env.yml
conda activate covid_vax_network
```

# Tests

After installing all dependencies, you can test your setup by executing the provided tests:

```bash
pytest
```

If you want to access code coverage statistics then execute:

```bash
pytest --cov
```

# Usage

If you use our code or the Twitter data set that we collected on Covid vaccination, please cite our paper:

```
@conference{béres2021vaccine,
  author       = {Ferenc Béres and Rita Csoma and Tamás Vilmos Michaletzky and András A. Benczúr}, 
  title        = {Vaccine skepticism detection by network embedding},
  booktitle    = {Book of Abstracts of the 10th International Conference on Complex Networks and Their Applications},
  pages        = {241--243},
  year         = {2021},
  isbn         = {978-2-9557050-5-6},
}
```

## 1. Data

To comply data publication policy of Twitter, we cannot share the raw data. Instead, we publish the generated feature components that we used to train models for the task of vaccine skepticism detection.
   
### Download data

We provide a [bash script](scripts/download_data.sh) (`download_data.sh`) to download our Twitter data set related to COVID19 vaccine skepticism.

```bash
./scripts/download_data.sh
```

The feature components are downloaded and decompressed into the `data` subfolder.

## 2. Experiments

We prepared a [script](scripts/train_eval_models.sh) to train and evaluate the models for every experiment.
Specify the following parameters to run the script:
   * **First argument:** specify the folder for generated feature components
   * **Second argument:** specify the target vaccine view for model training  (e.g. Pro-vaxxer, Vax-skeptic)
   * **Third argument:** Number of independent experiments to run for calculating model performance
   * **Fourth argument (optional):** Assign model training to a given GPU (e.g. cuda:1)
   * **Fifth argument (optional):** 

For example use, the following command to train models for vaccine skepticism detection:
```bash
cd scripts
bash train_eval_models.sh ../data/public_data_2022-10-27 Vax-skeptic 10
```

OR use a different command to detect pro-vaccine content:
```bash
cd scripts
bash train_eval_models.sh ../data/public_data_2022-10-27 Pro-vaxxer 10
```

Find the results for these experiments in the `data/public_data_2022-10-27/{Vax-skeptic_results OR Pro-vaxxer_results}`, respectively.

### Notes on model parameters

For reproducability we share a [JSON configuration file](http://info.ilab.sztaki.hu/~fberes/covid_vaccine_data/parameters.json) containing the parameters that we used to train our models.

Note that this file can be easily downloaded with the same [script](scripts/download_data.sh) that is provided for downloading our Twitter data set.

## 3. Node embedding

We provide a script to train your own node embedding model on the Twitter reply graph.

First, you need to preprocess the network. For example, in the command below we only use the first 100K edges and exclude nodes with less than 5 connections:
```bash
python ./scripts/node_embedding.py preprocess ./data/tweet_ids/reply_network.txt --con 3 --rows 100000
```

Next, train a node embedding model (e.g. DeepWalk) on the preprocessed network. The input file (3core_100000.csv) was created in the previous step.
```bash
python ./scripts/node_embedding.py fit 3core_100000.csv --model DeepWalk
```

The 5-dimensional user representations are exported to a CSV file in your working directory. 
```bash
head -3 DeepWalk_dim5_*.csv
```

The first column is the user identifier and the rest contains the representation for each node (Twitter user) of the reply network.
```
897201317223038976,-1.7705172,-2.9218996,3.7667134,-1.5700843,1.1719122
1280867032347664384,-3.7421749,-4.201378,1.2413,-2.1019526,2.6107976
1346645698176049152,-1.8614192,-3.9294648,1.6497265,-2.1258717,1.7651175
```
These user representations can be fed to the vaccine view classifier. However, training node embedding models on the whole Twitter reply network with millions of edges can take several days.

# Acknowledgements

Support by the the European Union project RRF-2.3.1-21-2022-00004 within the framework of the Artificial Intelligence National Laboratory

# What is coming to this repository?

- Scripts to donwload raw data from Twitter based on the provided Tweet identifiers
- Documentation to generate feature components from raw data
