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

for subset, modality in [
    ("xview", "joint"),
    ("xsub", "joint"),
]:
    subprocess.call(
        [
            "python3",
            "models/s_tr_mod/s_tr_mod.py",
            "--id",
            f"{DS_NAME}_{subset}_{modality}_train",
            "--gpus",
            "1",
            # "--distributed_backend",
            # "ddp",
            "--train",
            "--test",
            "--profile_model",
            "--max_epochs",
            "40",
            "--optimization_metric",
            "top1acc",
            "--batch_size",
            "12",
            "--num_workers",
            "10",
            "--dataset_normalization",
            "0",
            "--dataset_name",
            DS_NAME,
            "--dataset_classes",
            str(DS_PATH / "classes.yaml"),
            "--dataset_train_data",
            str(DS_PATH / subset / f"train_data_{modality}.npy"),
            "--dataset_val_data",
            str(DS_PATH / subset / f"val_data_{modality}.npy"),
            "--dataset_test_data",
            str(DS_PATH / subset / f"val_data_{modality}.npy"),
            "--dataset_train_labels",
            str(DS_PATH / subset / "train_label.pkl"),
            "--dataset_val_labels",
            str(DS_PATH / subset / "val_label.pkl"),
            "--dataset_test_labels",
            str(DS_PATH / subset / "val_label.pkl"),
            "--unfreeze_from_epoch",
            "0",
            "--unfreeze_layers_initial",
            "-1",
            "--learning_rate",
            "0.018",  # Linear scaling rule from lr 0.1 @ bs 64
            "--weight_decay",
            "0.0001",
            "--logging_backend",
            "wandb",
        ]
    )
