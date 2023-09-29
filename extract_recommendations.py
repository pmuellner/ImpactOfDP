from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
import numpy as np
import pickle as pl
import os
import torch
import gc
import argparse

def extract(path):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=path)

    uid_series = np.array([uid for token, uid in test_data.dataset.field2token_id[test_data.uid_field].items()
                           if token != "[PAD]"])

    # MultiVAE
    model = model.to(device=torch.device('cpu'))
    model.device = torch.device("cpu")
    model.history_item_id = model.history_item_id.to(device=torch.device('cpu'))
    model.history_item_value = model.history_item_value.to(device=torch.device('cpu'))

    topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=torch.device('cpu'))
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())

    recommendation_lists = dict()
    for idx, reclist in enumerate(external_item_list):
        internal_uid = uid_series[idx]
        external_uid = dataset.id2token(dataset.uid_field, internal_uid)
        recommendation_lists[external_uid] = reclist
    return recommendation_lists

LEN_SUFFIX = len(".pth")
dataset_name = "sportsandoutdoors"
model_name = "MultiVAE"
checkpoint_dir = "saved/" + dataset_name + "/" + model_name + "/"

parser = argparse.ArgumentParser()
parser.add_argument("--eps", required=True, type=str)
parser.add_argument("--nodp", default="true", type=str)
args = parser.parse_args()
epsilons = [eps for eps in args.eps.split(", ")]
if args.nodp == "true":
    args.nodp = True
else:
    args.nodp = False

if args.nodp:
    for saved_model_name in os.listdir(checkpoint_dir + "nodp"):
        if saved_model_name.endswith(".pkl"):
            continue
        recommendations = extract(checkpoint_dir + "nodp/" + saved_model_name)
        with open(checkpoint_dir + "nodp/" + saved_model_name[:-LEN_SUFFIX] + ".pkl", "wb") as f:
            pl.dump(recommendations, f)
        torch.cuda.empty_cache()
        gc.collect()

for eps in epsilons:
    for saved_model_name in os.listdir(checkpoint_dir + "e" + str(eps)):
        if saved_model_name.endswith(".pkl"):
            continue
        recommendations = extract(checkpoint_dir + "e" + str(eps) + "/" + saved_model_name)
        with open(checkpoint_dir + "e" + str(eps) + "/" + saved_model_name[:-LEN_SUFFIX] + ".pkl", "wb") as f:
            pl.dump(recommendations, f)
        torch.cuda.empty_cache()
        gc.collect()
