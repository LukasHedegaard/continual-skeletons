#  Benchmark the inference of all models

import os
import subprocess
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = Path(os.getenv("ROOT_PATH", default=""))
LOGS_PATH = Path(os.getenv("LOGS_PATH", default="logs"))
DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))

BATCH_SIZE = "1"
GPUS = "1"  # 0 if CPU

# Regular models
for model_name in ["a_gcn", "a_gcn_mod", "s_tr", "s_tr_mod", "st_gcn", "st_gcn_mod"]:
    subprocess.call(
        [
            "python3",
            f"models/{model_name}/{model_name}.py",
            "--id",
            f"benchmark_{'GPU' if GPUS=='1' else 'CPU'}",
            "--gpus",
            GPUS,
            "--profile_model",
            "--profile_model_num_runs",
            "100",
            "--batch_size",
            BATCH_SIZE,
            "--num_workers",
            BATCH_SIZE,
            "--dataset_name",
            "dummy",
            "--logging_backend",
            "wandb",
        ]
    )

# Continual models
for model_name in [
    "coa_gcn",
    "coa_gcn_mod",
    "cos_tr",
    "cos_tr_mod",
    "cost_gcn",
    "cost_gcn_mod",
]:
    subprocess.call(
        [
            "python3",
            f"models/{model_name}/{model_name}.py",
            "--id",
            f"benchmark_{'GPU' if GPUS=='1' else 'CPU'}",
            "--gpus",
            GPUS,
            "--profile_model",
            "--profile_model_num_runs",
            "100",
            "--forward_mode",
            "frame",
            "--batch_size",
            BATCH_SIZE,
            "--num_workers",
            BATCH_SIZE,
            "--dataset_name",
            "dummy",
            "--logging_backend",
            "wandb",
        ]
    )
