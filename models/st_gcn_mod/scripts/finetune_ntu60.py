import subprocess
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = Path(os.getenv("ROOT_PATH", default=""))
LOGS_PATH = Path(os.getenv("LOGS_PATH", default="logs"))
DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))

DS_NAME = "ntu"
DS_PATH = DATASETS_PATH / "ntu60"

for subset, modality, pretrained_model in [
    ("xview", "joint", "stgcn/nturgbd60_xview/ntu_cv_stgcn_joint-49-29400.pt"),
    ("xsub", "joint", "stgcn/nturgbd60_xsub/ntu_cs_stgcn_joint-49-31300.pt"),
]:
    subprocess.call(
        [
            "python3",
            "models/st_gcn_mod/st_gcn_mod.py",
            "--id",
            "finetune_all_layers",
            "--gpus",
            "1",
            "--train",
            "--max_epochs",
            "15",
            "--optimization_metric",
            "top1acc",
            "--test",
            "--batch_size",
            "12",
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
            str(ROOT_PATH / "pretrained_models" / pretrained_model),
            "--unfreeze_from_epoch",
            "0",
            "--unfreeze_layers_initial",
            "-1",
            "--learning_rate",
            "0.04",  # Linear scaling rule: 0,1 / 64 * 16 = 0.025
            "--weight_decay",
            "0.0001",
            "--finetune_from_weights",
            str(ROOT_PATH / "pretrained_models" / pretrained_model),
            "--logging_backend",
            "wandb",
        ]
    )
