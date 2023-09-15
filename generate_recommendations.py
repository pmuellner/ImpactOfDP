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
        "embedding_size": 64,
        "learning_rate": 0.0001,
        "dataset": "ml-1m",
        "model": "BPR",

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
    args = parser.parse_args()

    base_config["gpu_id"] = args.gpu_id

    for seed in range(5):
        new_config = base_config.copy()
        new_config["seed"] = seed
        new_config["checkpoint_dir"] = "saved/" + new_config["dataset"] + "/" + new_config["model"] + "/nodp/"
        if not os.path.exists(new_config["checkpoint_dir"]):
            os.makedirs(new_config["checkpoint_dir"])
        run_recbole(config_dict=new_config, saved=True)
        time.sleep(60)

    for eps in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10]:
        new_config = base_config.copy()
        new_config["benchmark_filename"] = ["train_e" + str(eps), "val", "test"]
        new_config["checkpoint_dir"] = "saved/" + new_config["dataset"] + "/" + new_config["model"] + "/e" + str(eps) + "/"
        if not os.path.exists(new_config["checkpoint_dir"]):
            os.makedirs(new_config["checkpoint_dir"])
        for seed in range(5):
            new_config["seed"] = seed
            run_recbole(config_dict=new_config, saved=True)
            time.sleep(60)

        #todo
        try:
            ram_info = psutil.virtual_memory()
            print(f"Total: {ram_info.total / 1024 / 1024 / 1024:.2f} GB")
            print(f"Available: {ram_info.available / 1024 / 1024 / 1024:.2f} GB")
            print(f"Used: {ram_info.used / 1024 / 1024 / 1024:.2f} GB")
            print(f"Percentage usage: {ram_info.percent}%")
        except FileNotFoundError:
            print("Ram info not available on this system")

        gc.collect()


