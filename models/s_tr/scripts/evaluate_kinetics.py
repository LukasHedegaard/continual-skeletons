import os
import subprocess
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = Path(os.getenv("ROOT_PATH", default=""))
LOGS_PATH = Path(os.getenv("LOGS_PATH", default="logs"))
DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))

GPUS = "1"
DS_NAME = "kinetics"
DS_PATH = DATASETS_PATH / DS_NAME

for modality, pretrained_model in [
    ("joint", "weights/str_kinetics_joint.ckpt"),
    ("bone", "weights/str_kinetics_bone.ckpt"),
]:
    subprocess.call(
        [
            "python3",
            "models/s_tr/s_tr.py",
            "--id",
            f"test_and_extract_{DS_NAME}_{modality}",
            "--gpus",
            GPUS,
            "--test",
            "--extract_features_after_layer",
            "fc",
            "--batch_size",
            "64",
            "--num_workers",
            "8",
            "--dataset_normalization",
            "0",
            "--dataset_name",
            DS_NAME,
            "--dataset_classes",
            str(DS_PATH / "classes.yaml"),
            "--dataset_train_data",
            str(DS_PATH / f"train_data_{modality}.npy"),
            "--dataset_val_data",
            str(DS_PATH / f"val_data_{modality}.npy"),
            "--dataset_test_data",
            str(DS_PATH / f"val_data_{modality}.npy"),
            "--dataset_train_labels",
            str(DS_PATH / "train_label.pkl"),
            "--dataset_val_labels",
            str(DS_PATH / "val_label.pkl"),
            "--dataset_test_labels",
            str(DS_PATH / "val_label.pkl"),
            "--finetune_from_weights",
            pretrained_model,
            "--logging_backend",
            "wandb",
        ]
    )
