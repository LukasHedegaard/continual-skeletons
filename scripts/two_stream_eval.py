import argparse
import pickle
from functools import reduce
from pathlib import Path
from typing import List

import numpy as np
import yaml
from ride import getLogger
from ride.metrics import topk_accuracies

logger = getLogger(__name__)


def load_labels(label_path: str):
    with open(label_path, "rb") as f:
        _, labels = pickle.load(f, encoding="latin1")

    return labels


def load_preds(path: str):
    path: Path = Path(path)
    assert path.exists(), f"{path} doesn't exist"
    assert "npy" in path.suffix, "Predictions should be stored as .npy files"

    # Load preds
    preds = np.load(path)
    return preds


def aggregate_preds(preds: List[np.array], method=np.add):
    assert method in {np.add, np.maximum}
    pred_shapes = [p.shape for p in preds]
    assert (
        len(set(pred_shapes)) == 1
    ), f"All preds should have the same shape but got {pred_shapes}"

    aggregated_preds = reduce(method, preds[1:], preds[0])

    return aggregated_preds


def multi_stream_eval(label_path: str, *pred_paths: str):
    targets = load_labels(label_path)
    preds = aggregate_preds(load_preds(p) for p in pred_paths if p)
    topks = [1, 3, 5]
    accs = topk_accuracies(preds, targets, topks)

    logger.info(yaml.dump({"Results": {f"top{k}acc": v for k, v in zip(topks, accs)}}))


if __name__ == "__main__":
    # construct the argument parser and parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--labels", type=str, required=True, help="Path to .pkl test labels."
    )
    parser.add_argument(
        "-p1", "--pred1", type=str, required=True, help="Path to .npy preds"
    )
    parser.add_argument(
        "-p2", "--pred2", type=str, required=True, help="Path to .npy preds"
    )
    parser.add_argument(
        "-p3", "--pred3", type=str, default="", help="Path to .npy preds"
    )
    parser.add_argument(
        "-p4", "--pred4", type=str, default="", help="Path to .npy preds"
    )

    args = parser.parse_args()
    multi_stream_eval(*args)
