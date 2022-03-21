import os
import subprocess
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = Path(os.getenv("ROOT_PATH", default=""))
DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))

DS_NAME = "kinetics"
DS_PATH = DATASETS_PATH / DS_NAME

subprocess.call(
    [
        "python3",
        "scripts/multi_stream_eval.py",
        "--log_as",
        f"CoAGcnMod/eval_{DS_NAME}_twostream",
        "--labels",
        str(DS_PATH / "val_label.pkl"),
        "--pred1",
        str(ROOT_PATH / "preds" / f"coagcnmod_{DS_NAME}_joint.npy"),
        "--pred2",
        str(ROOT_PATH / "preds" / f"coagcnmod_{DS_NAME}_bone.npy"),
    ]
)
