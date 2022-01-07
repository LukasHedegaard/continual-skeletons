import pytest
import torch

from models.a_gcn.a_gcn import AdaptiveGraphConvolution
from models.a_gcn_mod.a_gcn_mod import AGcnMod
from models.coa_gcn_mod.coa_gcn_mod import CoAGcnMod, TimeSlicedAdaptiveGraphConvolution
from datasets.ntu_rgbd import graph
import continual as co

def test_TimeSlicedAdaptiveGraphConvolution():
    SEQ_LEN=4
    BATCH_SIZE=2
    VERTICES=25
    IN_CHANNELS=3
    OUT_CHANNELS=4
    A=graph.A

    agc = co.forward_stepping(AdaptiveGraphConvolution(IN_CHANNELS, OUT_CHANNELS, A))
    
    tsagc = TimeSlicedAdaptiveGraphConvolution(IN_CHANNELS, OUT_CHANNELS, A)

    #  Transfer weights
    state_dict = agc.state_dict()
    tsagc.load_state_dict(state_dict)

    # Set to eval mode (otherwise batchnorm doesn't match)
    agc.eval()
    tsagc.eval()

    # Prepare sample
    sample_all = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN, VERTICES)
    sample_step0 = sample_all[:,:,0]
    sample_step1 = sample_all[:,:,1]

    # Reg
    all_steps_agc = agc.forward(sample_all)
    step0_agc = agc.forward_step(sample_step0)
    step1_agc = agc.forward_step(sample_step1)

    assert torch.allclose(all_steps_agc[:,:,0], step0_agc, atol=1e-5)
    assert torch.allclose(all_steps_agc[:,:,1], step1_agc, atol=1e-5)

    # TimeSliced
    all_steps_tsagc = tsagc.forward(sample_all)
    step0_tsagc = tsagc.forward_step(sample_step0)
    step1_tsagc = tsagc.forward_step(sample_step1)

    assert torch.equal(all_steps_tsagc[:,:,0], step0_tsagc)  # Equal!
    assert torch.equal(all_steps_tsagc[:,:,1], step1_tsagc)  # Equal!

    # Compare regular with time-sliced
    assert torch.allclose(step0_agc, step0_tsagc, atol=5e-4)
    assert torch.allclose(step1_agc, step1_tsagc, atol=5e-4)
    assert torch.allclose(all_steps_agc, all_steps_tsagc, atol=5e-4)



def dummy_hparams():
    d = CoAGcnMod.configs().default_values()
    d["max_epochs"] = 1
    d["batch_size"] = 2
    d["dataset_name"] = "dummy"
    d["accumulate_grad_batches"] = 1
    return d


def test_AGcnMod_dummy_params():
    # Model definition
    hparams = dummy_hparams()
    reg = AGcnMod(hparams)
    co = CoAGcnMod(hparams)

    #  Transfer weights
    co.load_state_dict(reg.state_dict())

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    # Prepare sample
    sample = torch.randn(reg.hparams.batch_size, *reg.input_shape)

    # Forward
    o_reg = reg(sample)
    o_co1 = co.forward(sample)
    o_co2 = co.forward(sample)

    assert torch.allclose(o_reg, o_co1, rtol=5e-4)
    assert torch.allclose(o_reg, o_co2, rtol=1e-4)
    assert torch.allclose(o_co1, o_co2, rtol=1e-4)


def real_hparams():
    d = CoAGcnMod.configs().default_values()
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
    ] = "models/a_gcn_mod/weights/agcnmod_ntu60_xview_joint.ckpt"
    return d


@pytest.mark.skip(reason="Model weights and dataset not available in CI/CD system")
def test_AGcnMod_real_params():
    # Model definition
    hparams = real_hparams()
    reg = AGcnMod(hparams)
    co = CoAGcnMod(hparams)

    #  Transfer weights
    co.load_state_dict(reg.state_dict())

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    # Prepare sample
    sample = next(iter(reg.train_dataloader()))[0]

    # Register hooks
    reg_features = []
    co_features = []

    def reg_store_features(sself, input, output):
        nonlocal reg_features
        reg_features = input

    def co_store_features(sself, input, output):
        nonlocal co_features
        co_features.append(input)

    reg.layers.layer1.gcn.soft.register_forward_hook(reg_store_features)
    co.layers.layer1.gcn.soft.register_forward_hook(co_store_features)

    # Forward
    reg(sample)
    co(sample)

    # Match for reg =----=
    #           co  ==----
    # where '=' is delay
    delay = 0
    checks = [
        torch.allclose(
            reg_features[0][:, :, i], co_features[i + delay][0].squeeze(), atol=5e-5
        )
        for i in range(delay, len(co_features) - delay)
    ]
    assert all(checks)

    # Conclusion: The matmul(A1, A2) yields an attn matrix over all vertices
    # Since it accounts for all timesteps, it cannot be made continual :-(
    # A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V


