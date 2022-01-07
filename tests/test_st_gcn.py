import torch

from models.st_gcn import StGcn


def default_hparams():
    d = StGcn.configs().default_values()
    d["max_epochs"] = 1
    d["batch_size"] = 2
    d["accumulate_grad_batches"] = 1
    return d


def test_forward():
    hparams = default_hparams()
    hparams["dataset_name"] = "dummy"
    net = StGcn(hparams)

    input = torch.rand((hparams["batch_size"], *net.input_shape))
    output = net(input)

    assert output.shape == (hparams["batch_size"], *net.output_shape)
