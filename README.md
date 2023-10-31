# The Impact of Differential Privacy on Recommendation Accuracy and Popularity Bias
Under Review.

## Abstract
Collaborative filtering-based recommender systems leverage vast amounts of behavioral user data, which poses severe privacy risks. Thus, often random noise is added to the data to ensure Differential Privacy (DP). However, to date it is not well understood in which ways this impacts personalized recommendations. In this work, we study how DP affects recommendation accuracy and popularity bias when applied to the training data of state-of-the-art recommendation models. Our findings are three-fold: First, we observe that nearly all users' recommendations change when DP is applied. Second, recommendation accuracy drops substantially while recommended item popularity experiences a sharp increase, suggesting that popularity bias worsens. Third, we find that DP exacerbates popularity bias more severely for users who prefer unpopular items than for users who prefer popular items.

## Requirements
* Python 3
* PyTorch
* RecBole
* NumPy
* Pands
* Pickle
* Matplotlib


## Instructions
Following steps need to be followed to reproduce our experiments

1. <i>Dataset Preprocessing</i>: Scaling listening events to a range of 1 to 5 (LastFM User Groups), apply user-core pruning (all datasets), apply item-core pruning (LastFM User Groups), filter positive feedback data.
```
dataset/preprocessing.ipynb
```

2. <i>Data Splitting and Applying DP</i>: Split datasets into 60% training data, 20% validation data, and 20% test data, generate multiple training datasets with DP (via using different $\epsilon$ values).
```
python data_splits_and_dp.py
```

3. <i>Hyperparameter Tuning</i>: Test various hyperparameter configurations for all models and datasets, identify the best parameters for a given model and dataset. More details are given in the RecBole documentation (https://recbole.io/docs/user_guide/usage/parameter_tuning.html). For example, to perform hyperparameter tuning on the MovieLens 1M dataset for MultVAE (hp/multvae_params contains the parameter configurations to be tested, and hp/multvae_config is the configuration file for RecBole):
```
python hyperparameter_tuning.py --dataset ml-1m --model_name MultiVAE --params hp/multivae_params --config hp/multivae_config
```

4. <i>Model Training</i>: Train a recommendation model on data with or without DP and use different random seeds. For example, to train a model (defined in the .py file) using GPU 1 different $\epsilon$ values and random seeds:
```
python train_models.py --gpu_id 1 --eps "0.1, 1, 3, nodp" --seeds "0, 1, 2, 3, 4"
```

5. <i>Evaluation</i>: Generate recommendation lists based on the trained models and save the recommendation lists to the disk (extract_recommendations.py). Then, evaluate the recommendations using the respective notebook (evaluate_recommendations.ipynb ):
```
python extract_recommendations.py
evaluate_recommendations.ipynb 
```


## Data
For this work, we only use datasets that are publicly available:
* MovieLens 1M [1] (https://grouplens.org/datasets/movielens/1m/)
* LastFM User Groups [2] (https://doi.org/10.5281/zenodo.3475975)
* Amazon Grocery and Gourmet Food [3] (https://nijianmo.github.io/amazon/index.html).

## References
[1] Harper, F. M., Konstan, J. A.: The movielens datasets: History and context. ACM Transactions on Interactive Intelligent Systems (TIIS) 5(4), 1–19 (2015)   

[2] Kowald, D., Schedl, M., Lex, E.: The unfairness of popularity bias in music reommendation: a reproducibility study. Advances in Information Retrieval 12036, 35 (2020)

[3] Ni, J., Li, J., McAuley, J.: Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In EMNLP-IJCNLP, pp. 188–197 (2019)