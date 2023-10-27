import pandas as pd
import numpy as np
import argparse
import os

np.random.seed(42)

# Definition of DP + sampling: Practical Privacy Preserving POI Recommendations
def apply_dp(pos_items, all_items, privacy_budget):
    prob_pos = 1 / (np.exp(privacy_budget) + 1) + (np.exp(privacy_budget) - 1) / (np.exp(privacy_budget) + 1)
    pos_from_pos_items = np.random.choice(list(pos_items),
                                          size=np.random.binomial(len(pos_items), p=prob_pos),
                                          replace=False)

    prob_neg = 1 / (np.exp(privacy_budget) + 1)
    all_neg_items = list(set(all_items).difference(pos_items))
    neg_items = np.random.choice(all_neg_items, size=min(len(pos_items), len(all_neg_items)), replace=False)
    pos_from_neg_items = np.random.choice(list(neg_items),
                                          size=np.random.binomial(len(neg_items), p=prob_neg),
                                          replace=False)

    new_pos_items = list(pos_from_pos_items) + list(pos_from_neg_items)
    return new_pos_items


DATASET = "grocery"
#positive_feedback_df = pd.read_csv("dataset/ml1m_preprocessed.csv", header=None, names=["user_id", "item_id"])
#positive_feedback_df = pd.read_csv("dataset/sportsoutdoors_preprocessed.csv", header=None, names=["user_id", "item_id"])
#positive_feedback_df = pd.read_csv("dataset/lfm3k_preprocessed.csv", header=None, names=["user_id", "item_id"])
positive_feedback_df = pd.read_csv("dataset/grocery_preprocessed.csv", header=None, names=["user_id", "item_id"])
positive_feedback_df.to_csv("dataset/" + DATASET + "/" + DATASET + ".inter", sep="\t", index=False)
print(positive_feedback_df["user_id"].nunique())

print(len(positive_feedback_df))

positive_feedback_df = positive_feedback_df.sample(frac=1)
positive_feedback = positive_feedback_df.groupby("user_id")["item_id"].apply(set)


feedback_train, feedback_val, feedback_test = dict(), dict(), dict()
profile_size = positive_feedback.apply(len)
for user_id, n in profile_size.items():
    # splits: 60% trainset, 20% valset, 20% testset
    # at least 1 feedback for train/val/test
    if n >= 3:
        n_val = int(np.ceil(n * 0.2))
        n_test = int(np.ceil(n * 0.2))
        n_train = n - n_val - n_test

        all_feedback = list(positive_feedback[user_id])
        np.random.shuffle(all_feedback)

        feedback_val[user_id] = all_feedback[:n_val]
        feedback_test[user_id] = all_feedback[n_val:n_val+n_test]
        feedback_train[user_id] = all_feedback[n_val+n_test:]

feedback_train = pd.Series(feedback_train)
feedback_val = pd.Series(feedback_val)
feedback_test = pd.Series(feedback_test)

# save splits to disk
train_df = feedback_train.explode().to_frame("item_id:token")
train_df["user_id:token"] = train_df.index
train_df.to_csv("dataset/" + DATASET + "/" + DATASET + ".train.inter", sep="\t", index=False)
all_items = set(train_df["item_id:token"].unique())

val_df = feedback_val.explode().to_frame("item_id:token")
val_df["user_id:token"] = val_df.index
val_df.to_csv("dataset/" + DATASET + "/" + DATASET + ".val.inter", sep="\t", index=False)

test_df = feedback_test.explode().to_frame("item_id:token")
test_df["user_id:token"] = test_df.index
test_df.to_csv("dataset/" + DATASET + "/" + DATASET + ".test.inter", sep="\t", index=False)

# generate dp datasets
epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10][::-1]
for eps in epsilons:
    train_dp = feedback_train.apply(lambda pos_items: apply_dp(pos_items, all_items, privacy_budget=eps))
    train_dp_df = train_dp.explode().to_frame("item_id:token")

    print(eps, len(train_dp_df))

    train_dp_df["user_id:token"] = train_dp_df.index
    train_dp_df.to_csv("dataset/" + DATASET + "/" + DATASET + ".train_e" + str(eps) + ".inter", sep="\t", index=False)


#%%
