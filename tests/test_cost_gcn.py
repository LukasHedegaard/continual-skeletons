from collections import OrderedDict

import torch

import continual
from datasets import ntu_rgbd
from models.base import (
    CoSpatioTemporalBlock,
    CoTemporalConvolution,
    SpatioTemporalBlock,
    TemporalConvolution,
)
from models.cost_gcn.cost_gcn import CoStGcn
from models.st_gcn.st_gcn import StGcn


def default_hparams():
    d = CoStGcn.configs().default_values()
    d["max_epochs"] = 1
    d["batch_size"] = 2
    d["accumulate_grad_batches"] = 1
    return d


def test_forward_shapes():
    hparams = default_hparams()
    hparams["dataset_name"] = "dummy"
    net = CoStGcn(hparams)

    sample = next(iter(net.train_dataloader()))
    output = net.forward(sample)

    assert output.shape == (hparams["batch_size"], *net.output_shape)


def test_CoTemporalConvolution():
    in_channels = 4
    out_channels = 4
    kernel_size = 9
    stride = 1
    padding = 4

    reg = TemporalConvolution(in_channels, out_channels, kernel_size, stride, padding)
    co = CoTemporalConvolution(in_channels, out_channels, kernel_size, padding, stride)

    # Transfer weights
    state_dict = reg.state_dict()
    co.load_state_dict(state_dict)

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    # Prepare data
    T = 20
    num_nodes = ntu_rgbd.graph.A.shape[-1]
    sample = torch.rand((2, in_channels, T, num_nodes))

    target = reg.forward(sample)

    # forward
    output = reg.forward(sample)
    assert torch.allclose(target, output)

    # forward_steps
    output = co.forward_steps(sample, pad_end=True)
    assert torch.allclose(target, output, atol=1e-6)


def test_CoSpatioTemporalBlock_no_residual():
    in_channels = 4
    out_channels = 4
    A = ntu_rgbd.graph.A
    stride = 1
    residual = False
    temporal_padding = 4

    reg = SpatioTemporalBlock(
        in_channels,
        out_channels,
        A,
        stride,
        residual,
        temporal_padding=temporal_padding,
    )
    co = CoSpatioTemporalBlock(
        in_channels,
        out_channels,
        A,
        stride,
        residual,
        padding=temporal_padding,
    )

    #  Transfer weights
    state_dict = reg.state_dict()
    co.load_state_dict(state_dict, strict=True)

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    # Prepare data
    kernel_size = 9
    T = 20
    B = 2
    num_nodes = ntu_rgbd.graph.A.shape[-1]
    sample = torch.rand((B, in_channels, T, num_nodes))

    # Forward
    target = reg(sample)

    # Frame-by-frame
    output = []
    for i in range(T):
        output.append(co.forward_step(sample[:, :, i]))

    checks = [
        torch.allclose(
            target[:, :, t],
            output[t + (kernel_size - 1 - reg.tcn.padding)],
            atol=5e-7,
        )
        for t in range(reg.tcn.padding, T - (kernel_size - 1))
    ]

    assert all(checks)


def test_CoSpatioTemporalBlock_residual_eq_channels():
    in_channels = 4
    out_channels = 4
    A = ntu_rgbd.graph.A
    stride = 1
    residual = True

    reg = SpatioTemporalBlock(in_channels, out_channels, A, stride, residual)
    co = CoSpatioTemporalBlock(
        in_channels, out_channels, A, stride, residual, padding=4
    )

    #  Transfer weights
    state_dict = reg.state_dict()
    state_dict = OrderedDict([("0.1." + k, v) for k, v in state_dict.items()])
    co.load_state_dict(state_dict)

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    # Prepare data
    kernel_size = 9
    T = 20
    B = 2
    num_nodes = ntu_rgbd.graph.A.shape[-1]
    sample = torch.rand((B, in_channels, T, num_nodes))

    # Forward
    target = reg(sample)

    # Frame-by-frame
    output = []
    for i in range(T):
        output.append(co.forward_step(sample[:, :, i]))

    checks = [
        torch.allclose(
            target[:, :, t],
            output[t + (kernel_size - 1 - reg.tcn.padding)],
            atol=5e-7,
        )
        for t in range(reg.tcn.padding, T - (kernel_size - 1))
    ]

    assert all(checks)


def test_CoSpatioTemporalBlock_residual_neq_channels():
    in_channels = 2
    out_channels = 4
    A = ntu_rgbd.graph.A
    stride = 1
    residual = True

    reg = SpatioTemporalBlock(in_channels, out_channels, A, stride, residual)
    co = CoSpatioTemporalBlock(
        in_channels, out_channels, A, stride, residual, padding=4
    )

    #  Transfer weights
    state_dict = reg.state_dict()
    mapping = {
        "res": "0.0.",
        "gcn": "0.1.",
        "tcn": "0.1.",
    }
    state_dict = OrderedDict([(mapping[k[:3]] + k, v) for k, v in state_dict.items()])
    co.load_state_dict(state_dict)

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    # Prepare data
    kernel_size = 9
    T = 20
    B = 2
    num_nodes = ntu_rgbd.graph.A.shape[-1]
    sample = torch.rand((B, in_channels, T, num_nodes))

    # Forward
    target = reg(sample)

    # Frame-by-frame
    output = []
    for i in range(T):
        output.append(co.forward_step(sample[:, :, i]))

    checks = [
        torch.allclose(
            target[:, :, t],
            output[t + (kernel_size - 1 - reg.tcn.padding)],
            atol=5e-7,
        )
        for t in range(reg.tcn.padding, T - (kernel_size - 1))
    ]

    assert all(checks)


def test_CoSpatioTemporalBlock_residual_neq_channels_strided():
    # Prepare data
    in_channels = 2
    out_channels = 2
    A = ntu_rgbd.graph.A
    T = 20
    B = 2
    num_nodes = ntu_rgbd.graph.A.shape[-1]
    sample = torch.rand((B, in_channels, T, num_nodes))

    stride = 2
    residual = True

    reg = SpatioTemporalBlock(in_channels, out_channels, A, stride, residual)
    co = CoSpatioTemporalBlock(
        in_channels, out_channels, A, stride, residual, padding=4
    )

    #  Transfer weights
    state_dict = reg.state_dict()
    mapping = {
        "res": "0.0.",
        "gcn": "0.1.",
        "tcn": "0.1.",
    }
    state_dict = OrderedDict([(mapping[k[:3]] + k, v) for k, v in state_dict.items()])
    co.load_state_dict(state_dict, strict=True)

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    target = reg(sample)

    # forward
    output = co.forward(sample)
    assert torch.allclose(target, output, atol=1e-6)

    # forward_steps
    output = co.forward_steps(sample, pad_end=True)
    assert torch.allclose(target, output, atol=1e-6)


def test_simple_costgcn():
    # Prepare data
    T = 40
    B = 2
    in_channels = 3
    out_channels = 4
    num_nodes = ntu_rgbd.graph.A.shape[-1]
    A = ntu_rgbd.graph.A
    sample = torch.rand((B, in_channels, T, num_nodes))

    stride = 2

    reg = torch.nn.Sequential(
        SpatioTemporalBlock(in_channels, in_channels, A, residual=False),
        SpatioTemporalBlock(in_channels, in_channels, A),
        SpatioTemporalBlock(in_channels, out_channels, A, stride=stride),
    )
    co = continual.Sequential(
        CoSpatioTemporalBlock(in_channels, in_channels, A, residual=False, padding=4),
        CoSpatioTemporalBlock(in_channels, in_channels, A, padding=4),
        CoSpatioTemporalBlock(in_channels, out_channels, A, stride=stride, padding=4),
    )

    #  Transfer weights
    state_dict = reg.state_dict()
    mapping = {
        "res": "0.0.",
        "gcn": "0.1.",
        "tcn": "0.1.",
    }

    def map_fn(k):
        k = k[:2] + mapping[k[2:5]] + k[2:]
        if k[:5] == "0.0.1":
            k = "0" + k[5:]
        return k

    state_dict = OrderedDict([(map_fn(k), v) for k, v in state_dict.items()])
    co.load_state_dict(state_dict)

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    target = reg(sample)

    # forward
    output = co.forward(sample)
    assert torch.allclose(target, output, atol=1e-6)

    # forward_steps
    output = co.forward_steps(sample, pad_end=True)
    assert torch.allclose(target, output, atol=1e-6)


def test_costgcn():
    # Model definition
    hparams = default_hparams()
    hparams["dataset_name"] = "dummy"
    hparams["batch_size"] = 1
    reg = StGcn(hparams)
    co = CoStGcn(hparams)

    #  Transfer weights
    co.load_state_dict(reg.state_dict(), strict=False)

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    # Prepare sample
    # sample = torch.randn(reg.hparams.batch_size, *reg.input_shape)
    sample = next(iter(reg.train_dataloader()))

    target = reg(sample)

    # forward
    o_co1 = co.forward(sample)
    assert torch.allclose(target, o_co1, rtol=5e-5)

    # forward_steps
    o_co2 = co.forward_steps(sample, pad_end=True)
    assert torch.allclose(target, o_co2, rtol=5e-5)
