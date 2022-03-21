import os
import subprocess
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = Path(os.getenv("ROOT_PATH", default=""))
LOGS_PATH = Path(os.getenv("LOGS_PATH", default="logs"))
DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))

GPUS = int(os.getenv("GPUS", default="1"))
BATCH_SIZE = 8
# Adjust LR using linear scaling rule
LEARNING_RATE = 0.01 / 128 * BATCH_SIZE * GPUS

DS_NAME = "kinetics"
DS_PATH = DATASETS_PATH / "kinetics"

for modality, pretrained_model in [
    ("joint", "weights/str_kinetics_joint.ckpt"),
    ("bone", "weights/str_kinetics_bone.ckpt"),
]:
    subprocess.call(
        [
            "python3",
            "models/s_tr/s_tr.py",
            "--id",
            f"{DS_NAME}_{modality}_train",
            "--gpus",
            str(GPUS),
            "--train",
            "--test",
            "--max_epochs",
            "20",
            "--optimization_metric",
            "top1acc",
            "--batch_size",
            str(BATCH_SIZE),
            "--num_workers",
            str(BATCH_SIZE // 2),
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
            "--unfreeze_from_epoch",
            "0",
            "--unfreeze_layers_initial",
            "-1",
            "--learning_rate",
            str(LEARNING_RATE),
            "--weight_decay",
            "0.0001",
            "--logging_backend",
            "wandb",
            "--accelerator",
            "ddp" if GPUS > 1 else "",
        ]
    )
