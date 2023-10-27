# The Impact of Differential Privacy on Recommendation Accuracy and Popularity Bias
Under Review.

## Abstract
Collaborative filtering-based recommender systems leverage vast amounts of behavioral user data, which poses severe privacy risks. Thus, often random noise is added to the data to ensure Differential Privacy (DP). However, to date it is not well understood in which ways this impacts personalized recommendations. In this work, we study how DP affects recommendation accuracy and popularity bias when applied to the training data of state-of-the-art recommendation models. Our findings are three-fold: First, we observe that nearly all users' recommendations change when DP is applied. Second, recommendation accuracy drops substantially while recommended item popularity experiences a sharp increase, suggesting that popularity bias worsens. Third, we find that DP exacerbates popularity bias more severely for users who prefer unpopular items than for users who prefer popular items. In sum, we study the accuracy–privacy trade-off, and provide novel insights into the particular trade-off between privacy and popularity bias.

## Requirements
* Python 3
* PyTorch
* RecBole
* NumPy
* Pands
* Pickle
* Matplotlib


## Instructions
1. TODO
2. 
```
TODO
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