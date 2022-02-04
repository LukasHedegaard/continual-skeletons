#  Benchmark the inference of all models

import os
import subprocess
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = Path(os.getenv("ROOT_PATH", default=""))
LOGS_PATH = Path(os.getenv("LOGS_PATH", default="logs"))
DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))

DS_NAME = "ntu60"
DS_PATH = DATASETS_PATH / "ntu60"

BATCH_SIZE="1"
GPUS="1"  # 0 if CPU

SUBSET="xview"
MODALITY="joint"

# Regular models
for model_name in ["a_gcn", "a_gcn_mod", "s_tr", "s_tr_mod", "st_gcn", "st_gcn_mod"]:
    subprocess.call(
        [
            "python3",
            f"models/{model_name}/{model_name}.py",
            "--id",
            f"{DS_NAME}_benchmark_{'GPU' if GPUS=='1' else 'CPU'}",
            "--gpus",
            GPUS,
            "--profile_model",
            "--batch_size",
            BATCH_SIZE,
            "--num_workers",
            BATCH_SIZE ,
            "--dataset_name",
            DS_NAME,
            "--dataset_classes",
            str(DS_PATH / "classes.yaml"),
            "--dataset_train_data",
            str(DS_PATH / SUBSET / f"train_data_{MODALITY}.npy"),
            "--dataset_val_data",
            str(DS_PATH / SUBSET / f"val_data_{MODALITY}.npy"),
            "--dataset_test_data",
            str(DS_PATH / SUBSET / f"val_data_{MODALITY}.npy"),
            "--dataset_train_labels",
            str(DS_PATH / SUBSET / "train_label.pkl"),
            "--dataset_val_labels",
            str(DS_PATH / SUBSET / "val_label.pkl"),
            "--dataset_test_labels",
            str(DS_PATH / SUBSET / "val_label.pkl"),
            "--logging_backend",
            "wandb",
        ]
    )

# Continual models
for model_name in ["coa_gcn", "coa_gcn_mod", "cos_tr", "cos_tr_mod", "cost_gcn", "cost_gcn_mod"]:
    subprocess.call(
        [
            "python3",
            f"models/{model_name}/{model_name}.py",
            "--id",
            f"{DS_NAME}_benchmark_{'GPU' if GPUS=='1' else 'CPU'}",
            "--gpus",
            GPUS,
            "--profile_model",
            "--forward_mode",
            "frame",
            "--batch_size",
            BATCH_SIZE,
            "--num_workers",
            BATCH_SIZE ,
            "--dataset_name",
            DS_NAME,
            "--dataset_classes",
            str(DS_PATH / "classes.yaml"),
            "--dataset_train_data",
            str(DS_PATH / SUBSET / f"train_data_{MODALITY}.npy"),
            "--dataset_val_data",
            str(DS_PATH / SUBSET / f"val_data_{MODALITY}.npy"),
            "--dataset_test_data",
            str(DS_PATH / SUBSET / f"val_data_{MODALITY}.npy"),
            "--dataset_train_labels",
            str(DS_PATH / SUBSET / "train_label.pkl"),
            "--dataset_val_labels",
            str(DS_PATH / SUBSET / "val_label.pkl"),
            "--dataset_test_labels",
            str(DS_PATH / SUBSET / "val_label.pkl"),
            "--logging_backend",
            "wandb",
        ]
    )
