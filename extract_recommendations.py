
from recbole.utils.case_study import full_sort_topk
import numpy as np
import pickle as pl
import os
import torch
import gc
import argparse
from recbole.utils import init_seed, init_logger, get_model
from logging import getLogger
from recbole.data import create_dataset, data_preparation


def load_data_and_model_cpu(model_file):
    checkpoint = torch.load(model_file)
    config = checkpoint["config"]

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, _, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    torch.device("cpu")
    model = get_model(config["model"])(config, train_data._dataset).to(torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return model, dataset, test_data

def extract(path):
    # load all necessary resources to the CPU
    model, dataset, test_data = load_data_and_model_cpu(model_file=path)


    uid_series = np.array([uid for token, uid in test_data.dataset.field2token_id[test_data.uid_field].items()
                           if token != "[PAD]"])
    model = model.to(device=torch.device('cpu'))
    model.device = torch.device("cpu")

    # MultVAE requires additional resources to be transferred to the CPU
    #model.history_item_id = model.history_item_id.to(device=torch.device('cpu'))
    #model.history_item_value = model.history_item_value.to(device=torch.device('cpu'))

    # For efficiency, generate recommendation lists for batches of users
    batch_size = 100
    n_batches = np.ceil(len(uid_series) / batch_size)
    recommendation_lists = dict()
    idx = 0
    for bid, batch in enumerate(np.array_split(uid_series, n_batches)):
        print("batch %d/%d" % (bid+1, n_batches))
        topk_score, topk_iid_list = full_sort_topk(batch, model, test_data, k=10, device=torch.device('cpu'))
        external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())

        for reclist in external_item_list:
            internal_uid = uid_series[idx]
            external_uid = dataset.id2token(dataset.uid_field, internal_uid)
            recommendation_lists[external_uid] = reclist
            idx += 1
    return recommendation_lists


LEN_SUFFIX = len(".pth")
dataset_name = "ml-1m"
model_name = "ENMF"
checkpoint_dir = "saved/" + dataset_name + "/" + model_name + "/"

parser = argparse.ArgumentParser()
# a list containing for which epsilon values the recommendation lists should be extracted, e.g., "0.1, 1, 4"
parser.add_argument("--eps", default="1", type=str)
args = parser.parse_args()
epsilons = [eps for eps in args.eps.split(", ")]

for eps in epsilons:
    if eps == "nodp":
        for saved_model_name in os.listdir(checkpoint_dir + "nodp"):
            if saved_model_name.endswith(".pkl"):
                continue
            recommendations = extract(checkpoint_dir + "nodp/" + saved_model_name)
            with open(checkpoint_dir + "nodp/" + saved_model_name[:-LEN_SUFFIX] + ".pkl", "wb") as f:
                pl.dump(recommendations, f)
            torch.cuda.empty_cache()
            gc.collect()
    else:
        for saved_model_name in os.listdir(checkpoint_dir + "e" + str(eps)):
            if saved_model_name.endswith(".pkl"):
                continue
            recommendations = extract(checkpoint_dir + "e" + str(eps) + "/" + saved_model_name)
            with open(checkpoint_dir + "e" + str(eps) + "/" + saved_model_name[:-LEN_SUFFIX] + ".pkl", "wb") as f:
                pl.dump(recommendations, f)
            torch.cuda.empty_cache()
            gc.collect()
