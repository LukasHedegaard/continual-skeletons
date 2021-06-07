import torch
from models.a_gcn_mod.a_gcn_mod import AGcnMod
from models.coa_gcn_mod.coa_gcn_mod import CoAGcnMod


def default_hparams():
    d = CoAGcnMod.configs().default_values()
    d["max_epochs"] = 1
    d["batch_size"] = 2
    return d


def test_AGcnMod():
    # Model definition
    hparams = default_hparams()
    hparams["dataset_name"] = "dummy"
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
    o_co = co(sample)

    assert torch.allclose(o_reg, o_co)
