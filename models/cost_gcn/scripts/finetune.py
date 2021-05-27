import subprocess
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = Path(os.getenv("ROOT_PATH", default=""))
LOGS_PATH = Path(os.getenv("LOGS_PATH", default="logs"))
DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))

GPUS = "2"
DS_NAME = "ntu"
DS_PATH = DATASETS_PATH / DS_NAME
DS_SUBSET = "xview"
MODALITY = "joint"


subprocess.call(
    [
        "python3",
        "models/cost_gcn/cost_gcn.py",
        "--id",
        "finetune",
        "--gpus",
        GPUS,
        "--forward_mode",
        "clip",
        "--train",
        "--max_epochs",
        "4",
        "--optimization_metric",
        "top1acc",
        "--test",
        "--batch_size",
        "20",
        "--num_workers",
        "8",
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
        "--finetune_from_weights",
        str(
            ROOT_PATH
            / "pretrained_models"
            / "stgcn"
            / "nturgbd60_cv"
            / "ntu_cv_stgcn_joint-49-29400.pt"
        ),
        "--logging_backend",
        "wandb",
        "--pool_size",
        "55",
        "--unfreeze_from_epoch",
        "0",
        "--unfreeze_layers_initial",
        "1",
        "--unfreeze_layers_max",
        "1",
        "--learning_rate",
        "0.002",
        "--weight_decay",
        "0.0001",
        "--distributed_backend",
        "ddp",
        # "--auto_lr_find",
        # "learning_rate",
        # "--auto_scale_batch_size",
        # "power",
    ]
)
