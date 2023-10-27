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
        "model": "LightGCN",
        "embedding_size": 64,
        "learning_rate": 0.001,
        "n_layers": 4,
        "reg_weight": 0.01,

        "data_path": "dataset/",
        "benchmark_filename": ["train", "val", "test"],
        "train_batch_size": 4096,
        "epochs": 5000,
        "stopping_step": 50,
        "eval_args": {
            "mode": "full",
            "order": "RO",
            "split": None
        },
        #"train_neg_sample_args": None,
        "train_neg_sample_args": {
            "distribution": "uniform",
            "sample_num": 1,
            "dynamic": False
        },
        "metrics": ["Recall"],
        "repeatable": False,
        "shuffle": True,
        "topk": 10,
        "valid_metric": "Recall@10",
        "eval_batch_size": 1000000
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="0", required=False, type=str)
    parser.add_argument("--eps", required=True, type=str)
    parser.add_argument("--seeds", default="0", type=str)
    args = parser.parse_args()
    epsilons = [eps for eps in args.eps.split(", ")]
    seeds = [seed for seed in args.seeds.split(", ")]
    base_config["gpu_id"] = args.gpu_id

    for eps in epsilons:
        new_config = base_config.copy()
        if eps == "nodp":
            new_config["checkpoint_dir"] = "saved/" + new_config["dataset"] + "/" + new_config["model"] + "/nodp/"
        else:
            new_config["benchmark_filename"] = ["train_e" + str(eps), "val", "test"]
            new_config["checkpoint_dir"] = "saved/" + new_config["dataset"] + "/" + new_config["model"] + "/e" + str(eps) + "/"

        if not os.path.exists(new_config["checkpoint_dir"]):
            os.makedirs(new_config["checkpoint_dir"])
        for seed in seeds:
            print(seed)
            new_config["seed"] = int(seed)
            run_recbole(config_dict=new_config, saved=True)
            torch.cuda.empty_cache()
            gc.collect()
