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
DS_SUBSET = "xview"
MODALITY = "joint"

subprocess.call(
    [
        "python3",
        "models/cost_gcn/cost_gcn.py",
        "--id",
        "finetune_all_layers",
        "--gpus",
        "4",
        "--forward_mode",
        "clip",
        "--train",
        "--max_epochs",
        "20",
        "--optimization_metric",
        "top1acc",
        "--test",
        "--profile_model",
        "--batch_size",
        "18",
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
        "models/st_gcn/weights/stgcn_ntu60_xview_joint.pt",
        "--logging_backend",
        "wandb",
        "--pool_size",
        "56",
        "--unfreeze_from_epoch",
        "0",
        "--unfreeze_layers_initial",
        "-1",
        "--learning_rate",
        "0.4",  # Apply scaling rule: 0,1 / 64 * 18 * 4 (gpus) = 0,125
        "--weight_decay",
        "0.0001",
        "--distributed_backend",
        "ddp",
    ]
)


# subprocess.call(
#     [
#         "python3",
#         "models/cost_gcn/cost_gcn.py",
#         "--id",
#         "finetune_2_last_layers",
#         "--gpus",
#         "1",
#         "--forward_mode",
#         "clip",
#         "--train",
#         "--max_epochs",
#         "10",
#         "--optimization_metric",
#         "top1acc",
#         "--test",
#         "--batch_size",
#         "22",
#         "--num_workers",
#         "8",
#         "--dataset_normalization",
#         "0",
#         "--dataset_name",
#         DS_NAME,
#         "--dataset_classes",
#         str(DS_PATH / "classes.yaml"),
#         "--dataset_train_data",
#         str(DS_PATH / DS_SUBSET / f"train_data_{MODALITY}.npy"),
#         "--dataset_val_data",
#         str(DS_PATH / DS_SUBSET / f"val_data_{MODALITY}.npy"),
#         "--dataset_test_data",
#         str(DS_PATH / DS_SUBSET / f"val_data_{MODALITY}.npy"),
#         "--dataset_train_labels",
#         str(DS_PATH / DS_SUBSET / "train_label.pkl"),
#         "--dataset_val_labels",
#         str(DS_PATH / DS_SUBSET / "val_label.pkl"),
#         "--dataset_test_labels",
#         str(DS_PATH / DS_SUBSET / "val_label.pkl"),
#         "--finetune_from_weights",
#         str(
#             ROOT_PATH
#             / "pretrained_models"
#             / "stgcn"
#             / "nturgbd60_cv"
#             / "ntu_cv_stgcn_joint-49-29400.pt"
#         ),
#         "--logging_backend",
#         "wandb",
#         "--pool_size",
#         "56",
#         "--unfreeze_from_epoch",
#         "0",
#         "--unfreeze_layers_initial",
#         "2",
#         "--unfreeze_layers_max",
#         "2",
#         "--learning_rate",
#         "0.00034375",  # Apply scaling rule: 0,001 / 64 * 22
#         "--weight_decay",
#         "0.0001",
#     ]
# )

# subprocess.call(
#     [
#         "python3",
#         "models/cost_gcn/cost_gcn.py",
#         "--id",
#         "finetune_bn",
#         "--gpus",
#         "1",
#         "--forward_mode",
#         "clip",
#         "--train",
#         "--max_epochs",
#         "10",
#         "--optimization_metric",
#         "top1acc",
#         "--test",
#         "--batch_size",
#         "22",
#         "--num_workers",
#         "8",
#         "--dataset_normalization",
#         "0",
#         "--dataset_name",
#         DS_NAME,
#         "--dataset_classes",
#         str(DS_PATH / "classes.yaml"),
#         "--dataset_train_data",
#         str(DS_PATH / DS_SUBSET / f"train_data_{MODALITY}.npy"),
#         "--dataset_val_data",
#         str(DS_PATH / DS_SUBSET / f"val_data_{MODALITY}.npy"),
#         "--dataset_test_data",
#         str(DS_PATH / DS_SUBSET / f"val_data_{MODALITY}.npy"),
#         "--dataset_train_labels",
#         str(DS_PATH / DS_SUBSET / "train_label.pkl"),
#         "--dataset_val_labels",
#         str(DS_PATH / DS_SUBSET / "val_label.pkl"),
#         "--dataset_test_labels",
#         str(DS_PATH / DS_SUBSET / "val_label.pkl"),
#         "--finetune_from_weights",
#         str(
#             ROOT_PATH
#             / "pretrained_models"
#             / "stgcn"
#             / "nturgbd60_cv"
#             / "ntu_cv_stgcn_joint-49-29400.pt"
#         ),
#         "--logging_backend",
#         "wandb",
#         "--pool_size",
#         "56",
#         "--unfreeze_from_epoch",
#         "0",
#         "--unfreeze_layers_initial",
#         "-1",
#         "--unfreeze_layers_must_include",
#         "bn",
#         "--learning_rate",
#         "0.00034375",  # Apply scaling rule: 0,001 / 64 * 22
#         "--weight_decay",
#         "0.0001",
#     ]
# )
