#  Benchmark the inference of all models

import os
import subprocess
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = Path(os.getenv("ROOT_PATH", default=""))
LOGS_PATH = Path(os.getenv("LOGS_PATH", default="logs"))
DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))

DEVICE = "RTX2080Ti"
NUM_RUNS = "100"
BATCH_SIZE = {"CPU": 1, "RTX2080Ti": 128}[DEVICE]
GPUS = "0" if DEVICE == "CPU" else "1"

# Regular models
for model_name in [
    "st_gcn",
    "st_gcn_mod",
    "a_gcn",
    "a_gcn_mod",
    "s_tr",
    "s_tr_mod",
]:
    BS = str(max(1, BATCH_SIZE // 2 if "mod" in model_name else BATCH_SIZE))
    subprocess.call(
        [
            "python3",
            f"models/{model_name}/{model_name}.py",
            "--id",
            f"benchmark_{DEVICE}_ntu60",
            "--gpus",
            GPUS,
            "--profile_model",
            "--profile_model_num_runs",
            NUM_RUNS,
            "--batch_size",
            BS,
            "--num_workers",
            BS,
            "--dataset_name",
            "dummy_ntu",
            "--logging_backend",
            "wandb",
        ]
    )

BATCH_SIZE = str({"CPU": 1, "RTX2080Ti": 256}[DEVICE])

# Continual models
for model_name in [
    "cost_gcn",
    "cost_gcn_mod",
    "coa_gcn",
    "coa_gcn_mod",
    "cos_tr",
    "cos_tr_mod",
]:
    subprocess.call(
        [
            "python3",
            f"models/{model_name}/{model_name}.py",
            "--id",
            f"benchmark_{DEVICE}_ntu60",
            "--gpus",
            GPUS,
            "--profile_model",
            "--profile_model_num_runs",
            NUM_RUNS,
            "--forward_mode",
            "frame",
            "--batch_size",
            BATCH_SIZE,
            "--num_workers",
            BATCH_SIZE,
            "--dataset_name",
            "dummy_ntu",
            "--logging_backend",
            "wandb",
        ]
    )
