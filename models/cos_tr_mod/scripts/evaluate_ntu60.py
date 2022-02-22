import os
import subprocess
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = Path(os.getenv("ROOT_PATH", default=""))
LOGS_PATH = Path(os.getenv("LOGS_PATH", default="logs"))
DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))

GPUS = "1"
DS_NAME = "ntu60"
DS_PATH = DATASETS_PATH / DS_NAME


for subset, modality, pretrained_model in [
    ("xview", "joint", "weights/strmod_ntu60_xview_joint.ckpt"),
    ("xsub", "joint", "weights/strmod_ntu60_xsub_joint.ckpt"),
]:

    subprocess.call(
        [
            "python3",
            "models/cos_tr_mod/cos_tr_mod.py",
            "--id",
            f"eval_{DS_NAME}_{subset}_{modality}",
            "--gpus",
            GPUS,
            "--forward_mode",
            "frame",
            "--test",
            "--batch_size",
            "128",
            "--num_workers",
            "8",
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
            "--finetune_from_weights",
            pretrained_model,
            "--logging_backend",
            "wandb",
        ]
    )
