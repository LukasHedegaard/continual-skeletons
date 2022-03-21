import os
import subprocess
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = Path(os.getenv("ROOT_PATH", default=""))
DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))

DS_NAME = "ntu60"
DS_PATH = DATASETS_PATH / DS_NAME

for subset in ["xview", "xsub"]:
    subprocess.call(
        [
            "python3",
            "scripts/multi_stream_eval.py",
            "--log_as",
            f"CoSTr/eval_{DS_NAME}_{subset}_twostream",
            "--labels",
            str(DS_PATH / subset / "val_label.pkl"),
            "--pred1",
            str(ROOT_PATH / "preds" / f"costr_{DS_NAME}_{subset}_joint.npy"),
            "--pred2",
            str(ROOT_PATH / "preds" / f"costr_{DS_NAME}_{subset}_bone.npy"),
        ]
    )
