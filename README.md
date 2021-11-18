# Vaccine skepticism detection by network embedding

![build](https://github.com/ferencberes/covid-vaccine-network/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/ferencberes/covid-vaccine-network/branch/main/graph/badge.svg?token=B2D3JHO2K3)](https://codecov.io/gh/ferencberes/covid-vaccine-network)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/django)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ferencberes/cid-vaccine-network/HEAD?filepath=ipython%2FVaxxerModelResults.ipynb)

[Ferenc Béres](https://github.com/ferencberes), [Rita Csoma](https://github.com/csomarita), [Tamás Vilmos Michaletzky](https://github.com/tmichaletzky), and [András A. Benczúr](https://mi.nemzetilabor.hu/people/andras-benczur)


In this work, we intended to develop techniques that are able to efficiently differentiate between pro-vaxxer and vax-skeptic Twitter content related to COVID19 vaccines. After multiple data preprocessing steps, we evaluated Tweet content and user interaction network classification by combining text classifiers with several node embedding and community detection
models.

# Requirements

- UNIX environment
- Install Python dependencies with the following command:

```bash
pip install -r requirements.txt
```

# Usage


## 1. Data

To comply data publication policy of Twitter, we cannot share the raw data. Instead, we publish our data in two packages to provide reproducibility and encourage future work:

- **[1. Twitter data identifiers](http://info.ilab.sztaki.hu/~fberes/covid_vaccine_data/tweet_ids_2021-11-18.zip):** contains only tweet ID and user ID for each collected tweet. We further publish the underlying reply graph that we used to fit node embedding and community detection methods. 

- **[2. Tweet representations](http://info.ilab.sztaki.hu/~fberes/covid_vaccine_data/covid_vaxxer_representations_2021-09-24.zip):** In this package, we publish the data that we used for training and evaluating the vaccine view classifier. This model predicts the vaccine view (pro-vaxxer or vax-skeptic) for each tweet by incorporating three modalities:

   * **1. text:** 1,000 dimensional TF-IDF vector of tweet text;
   * **2. history:** Four basic statistics calculated from past tweet labels of the same user;
   * **3. embedding:** 128-dimensional user representation learned by node embedding from the Twitter reply network.
   
## Download data

We provide a bash script (`download_data.sh`) to download our Twitter data set related to COVID19 vaccine skepticism.

```bash
./scripts/download_data.sh
```

Both packages are downloaded and decompressed into the `data` subfolder.

## 2. Tests

After downloading the data, you can test your setup by executing the provided tests:

```bash
pytest --cov
```

## 3. Results

We present our experimental results in a [jupyter notebook](ipython/VaxxerModelResults.ipynb)
You can run this notebook with [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ferencberes/cid-vaccine-network/HEAD?filepath=ipython%2FVaxxerModelResults.ipynb).

# What's coming to this repository?

This repository is still under development. In the upcoming weeks, we will publish:
- scripts to download the raw data from Twitter
- scripts to clean and preprocess raw data
- detailed documentation
