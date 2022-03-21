import os
import subprocess
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = Path(os.getenv("ROOT_PATH", default=""))
LOGS_PATH = Path(os.getenv("LOGS_PATH", default="logs"))
DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))

GPUS = "1"
DS_NAME = "ntu120"
DS_PATH = DATASETS_PATH / DS_NAME

for subset, modality, pretrained_model in [
    ("xset", "joint", "weights/agcnmod_ntu120_xset_joint.ckpt"),
    ("xsub", "joint", "weights/agcnmod_ntu120_xsub_joint.ckpt"),
    ("xset", "bone", "weights/agcnmod_ntu120_xset_bone.ckpt"),
    ("xsub", "bone", "weights/agcnmod_ntu120_xsub_bone.ckpt"),
]:
    subprocess.call(
        [
            "python3",
            "models/coa_gcn_mod/coa_gcn_mod.py",
            "--id",
            f"test_and_extract_{DS_NAME}_{subset}_{modality}",
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
