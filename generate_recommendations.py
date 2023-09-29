from recbole.quick_start import run_recbole
import numpy as np
import argparse
import os
import glob
import torch
import time
import psutil
import gc

if __name__ == '__main__':
    base_config = {
        "dataset": "LFM-3k",
        "model": "Pop",

        "data_path": "dataset/",
        "benchmark_filename": ["train", "val", "test"],
        "train_batch_size": 1024,
        "epochs": 1000,
        "stopping_step": 10,
        "eval_args": {
            "mode": "full",
            "order": "RO"
        },
        "load_col": {
            "inter": ["user_id", "item_id", "rating"]
        },
        "metrics": ["Recall", "NDCG", "AveragePopularity", "TailPercentage"],
        "repeatable": "true",
        "topk": 10,
        "valid_metric": "Recall@10",
        "eval_batch_size": 1000000
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="0", required=False, type=str)
    parser.add_argument("--eps", required=True, type=str)
    parser.add_argument("--nodp", default="true", type=str)
    args = parser.parse_args()
    epsilons = [eps for eps in args.eps.split(", ")]
    if args.nodp == "true":
        args.nodp = True
    else:
        args.nodp = False

    base_config["gpu_id"] = args.gpu_id

    if args.nodp:
        for seed in range(5):
            new_config = base_config.copy()
            new_config["seed"] = seed
            new_config["checkpoint_dir"] = "saved/" + new_config["dataset"] + "/" + new_config["model"] + "/nodp/"
            if not os.path.exists(new_config["checkpoint_dir"]):
                os.makedirs(new_config["checkpoint_dir"])
            run_recbole(config_dict=new_config, saved=True)
            torch.cuda.empty_cache()
            gc.collect()

    for eps in epsilons:
        new_config = base_config.copy()
        new_config["benchmark_filename"] = ["train_e" + str(eps), "val", "test"]
        new_config["checkpoint_dir"] = "saved/" + new_config["dataset"] + "/" + new_config["model"] + "/e" + str(eps) + "/"
        if not os.path.exists(new_config["checkpoint_dir"]):
            os.makedirs(new_config["checkpoint_dir"])
        for seed in range(5):
            new_config["seed"] = seed
            run_recbole(config_dict=new_config, saved=True)
            torch.cuda.empty_cache()
            gc.collect()
