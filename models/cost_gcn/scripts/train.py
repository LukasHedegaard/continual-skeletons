import os
import subprocess
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = Path(os.getenv("ROOT_PATH", default=""))
LOGS_PATH = Path(os.getenv("LOGS_PATH", default="logs"))
DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))

DS_NAME = "ntu"
DS_PATH = DATASETS_PATH / "ntu60"
DS_SUBSET = "xview"
MODALITY = "joint"


subprocess.call(
    [
        "python3",
        "models/cost_gcn/cost_gcn.py",
        "--id",
        "train",
        "--gpus",
        "4",
        "--forward_mode",
        "clip",
        "--train",
        "--test",
        "--max_epochs",
        "30",
        "--optimization_metric",
        "top1acc",
        "--test",
        "--batch_size",
        "18",
        "--num_workers",
        "9",
        "--learning_rate",
        "0.12",
        "--weight_decay",
        "0.0001",
        "--dataset_normalization",
        "0",
        "--dataset_name",
        DS_NAME,
        "--dataset_classes",
        str(DS_PATH / "classes.yaml"),
        "--dataset_train_data",
        str(DS_PATH / DS_SUBSET / f"train_data_{MODALITY}.npy"),
        "--dataset_val_data",
        str(DS_PATH / DS_SUBSET / f"val_data_{MODALITY}.npy"),
        "--dataset_test_data",
        str(DS_PATH / DS_SUBSET / f"val_data_{MODALITY}.npy"),
        "--dataset_train_labels",
        str(DS_PATH / DS_SUBSET / "train_label.pkl"),
        "--dataset_val_labels",
        str(DS_PATH / DS_SUBSET / "val_label.pkl"),
        "--dataset_test_labels",
        str(DS_PATH / DS_SUBSET / "val_label.pkl"),
        "--logging_backend",
        "wandb",
        "--distributed_backend",
        "ddp",
    ]
)
