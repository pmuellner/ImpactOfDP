from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
import numpy as np
import pickle as pl
import os
import torch

# todo read all models in folder
# extract recommendations
# keep folder structure
# recommendations file should have the same name as model
#
def extract(path):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=path)

    uid_series = np.array([uid for token, uid in test_data.dataset.field2token_id[test_data.uid_field].items()
                           if token != "[PAD]"])

    print(torch.cuda.is_available())
    """if torch.cuda.is_available():
        print("GPU")
        topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=torch.device('cuda'))
    else:
        print("CPU")
        topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=torch.device('cpu'))"""
    topk_score, topk_iid_list = full_sort_topk(uid_series, model.cpu(), test_data, k=10, device=torch.device('cpu'))
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())

    recommendation_lists = dict()
    for idx, reclist in enumerate(external_item_list):
        internal_uid = uid_series[idx]
        external_uid = dataset.id2token(dataset.uid_field, internal_uid)
        recommendation_lists[external_uid] = list(map(int, reclist))

    return recommendation_lists

import torch
torch.cuda.empty_cache()

LEN_SUFFIX = len(".pth")
dataset_name = "ml-1m"
model_name = "MultiVAE"
checkpoint_dir = "saved/" + dataset_name + "/" + model_name + "/"

for saved_model_name in os.listdir(checkpoint_dir + "nodp"):
    if saved_model_name.endswith(".pkl"):
        continue
    recommendations = extract(checkpoint_dir + "nodp/" + saved_model_name)
    with open(checkpoint_dir + "nodp/" + saved_model_name[:-LEN_SUFFIX] + ".pkl", "wb") as f:
        pl.dump(recommendations, f)

for eps in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10]:
    for saved_model_name in os.listdir(checkpoint_dir + "e" + str(eps)):
        if saved_model_name.endswith(".pkl"):
            continue
        recommendations = extract(checkpoint_dir + "e" + str(eps) + "/" + saved_model_name)
        with open(checkpoint_dir + "e" + str(eps) + "/" + saved_model_name[:-LEN_SUFFIX] + ".pkl", "wb") as f:
            pl.dump(recommendations, f)
