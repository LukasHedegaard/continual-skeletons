from collections import OrderedDict

import pytest
import torch

from datasets import ntu_rgbd
from models.base import CoSpatioTemporalBlock, SpatioTemporalBlock
from models.cost_gcn_mod.cost_gcn_mod import CoStGcnMod
from models.st_gcn_mod.st_gcn_mod import StGcnMod


def test_StGcnModBlock_residual_eq_channels():
    in_channels = 4
    out_channels = 4
    A = ntu_rgbd.graph.A
    stride = 1
    residual = True
    padding = 4

    reg = SpatioTemporalBlock(
        in_channels, out_channels, A, stride, residual, temporal_padding=padding
    )
    mod = SpatioTemporalBlock(
        in_channels, out_channels, A, stride, residual, temporal_padding=0
    )
    co = CoSpatioTemporalBlock(in_channels, out_channels, A, stride, residual)

    #  Transfer weights
    state_dict = reg.state_dict()
    mod.load_state_dict(state_dict)

    state_dict = OrderedDict([("0.1." + k, v) for k, v in state_dict.items()])
    co.load_state_dict(state_dict, strict=True)

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    mod.eval()
    co.eval()

    # Prepare data
    T = 20
    B = 2
    num_nodes = ntu_rgbd.graph.A.shape[-1]
    sample = torch.rand((B, in_channels, T, num_nodes))

    # Forward
    target = reg.forward(sample)

    # Mod
    mod_target = mod.forward(sample)
    assert torch.allclose(target[:, :, padding:-padding], mod_target)

    # Frame-by-frame
    output = co.forward(sample)
    assert torch.allclose(target[:, :, padding:-padding], output)


def dummy_hparams():
    d = CoStGcnMod.configs().default_values()
    d["max_epochs"] = 1
    d["batch_size"] = 2
    d["dataset_name"] = "dummy"
    d["accumulate_grad_batches"] = 1
    return d


def test_StGcnMod_dummy_params():
    # Model definition
    hparams = dummy_hparams()
    reg = StGcnMod(hparams)
    co = CoStGcnMod(hparams)

    #  Transfer weights
    co.load_state_dict(reg.state_dict(), strict=True)

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    # Prepare sample
    sample = torch.randn(reg.hparams.batch_size, *reg.input_shape)

    o_reg = reg.forward(sample)

    # Forward
    o_co1 = co.forward(sample)
    co.clean_state()
    o_co2 = co.forward_steps(sample)

    assert torch.allclose(o_co1, o_co2, atol=5e-4)
    assert torch.allclose(o_reg, o_co1, atol=5e-4)


def real_hparams():
    d = CoStGcnMod.configs().default_values()
    d["max_epochs"] = 1
    d["batch_size"] = 2

    import os
    from pathlib import Path

    DS_NAME = "ntu60"
    DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))
    DS_PATH = DATASETS_PATH / DS_NAME
    d["dataset_name"] = DS_NAME
    subset = "xview"
    modality = "joint"
    d["dataset_name"] = DS_NAME
    d["dataset_classes"] = str(DS_PATH / "classes.yaml")
    d["dataset_train_data"] = str(DS_PATH / subset / f"train_data_{modality}.npy")
    d["dataset_val_data"] = str(DS_PATH / subset / f"val_data_{modality}.npy")
    d["dataset_test_data"] = str(DS_PATH / subset / f"val_data_{modality}.npy")
    d["dataset_train_labels"] = str(DS_PATH / subset / "train_label.pkl")
    d["dataset_val_labels"] = str(DS_PATH / subset / "val_label.pkl")
    d["dataset_test_labels"] = str(DS_PATH / subset / "val_label.pkl")
    d[
        "finetune_from_weights"
    ] = "models/st_gcn_mod/weights/stgcnmod_ntu60_xview_joint.ckpt"
    return d


@pytest.mark.skip(reason="Model weights and dataset not available in CI/CD system")
def test_StGcnMod_real_params():
    # Model definition
    hparams = real_hparams()
    reg = StGcnMod(hparams)
    co = CoStGcnMod(hparams)

    #  Transfer weights
    co.load_state_dict(reg.state_dict())

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    # Prepare sample
    # sample = torch.randn(reg.hparams.batch_size, *reg.input_shape)
    sample = next(iter(reg.train_dataloader()))[0]

    # Forward
    o_co1 = co.forward(sample)
    o_co2 = co.forward_steps(sample)
    o_reg = reg(sample)

    assert torch.allclose(o_reg, o_co1, rtol=5e-5)
    assert torch.allclose(o_reg, o_co2, rtol=5e-5)
