import torch
from models.cost_gcn.cost_gcn import CoStGcn, CoStGcnBlock, CoTemporalConvolution
from models.st_gcn.st_gcn import StGcnBlock, TemporalConvolution
from datasets import ntu_rgbd
import numpy as np


def default_hparams():
    d = CoStGcn.configs().default_values()
    d["max_epochs"] = 1
    d["batch_size"] = 2
    return d


def test_forward_shapes():
    hparams = default_hparams()
    hparams["dataset_name"] = "dummy"
    net = CoStGcn(hparams)

    input = torch.rand((hparams["batch_size"], *net.input_shape))
    output = net(input)

    assert output.shape == (hparams["batch_size"], *net.output_shape)


def test_CoTemporalConvolution():
    in_channels = 4
    out_channels = 4
    kernel_size = 9
    stride = 1

    reg = TemporalConvolution(in_channels, out_channels, kernel_size, stride)
    co = CoTemporalConvolution(
        in_channels, out_channels, kernel_size, stride, extra_delay=None
    )

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

    # Forward
    target = reg(sample)

    # Frame-by-frame
    output = []
    for i in range(T):
        output.append(co(sample[:, :, i]))

    # Match after delay of T - 1
    checks = [
        torch.allclose(
            target[:, :, t],
            output[t + (kernel_size - 1 - reg.pad)],
            atol=5e-7,
        )
        for t in range(reg.pad, sample.shape[2] - (kernel_size - 1))
    ]

    assert all(checks)


def test_CoStGcnBlock_no_residual():
    in_channels = 4
    out_channels = 4
    A = ntu_rgbd.graph.A
    stride = 1
    residual = False

    reg = StGcnBlock(in_channels, out_channels, A, stride, residual)
    co = CoStGcnBlock(in_channels, out_channels, A, stride, residual)

    #  Transfer weights
    state_dict = reg.state_dict()
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
        output.append(co(sample[:, :, i]))

    checks = [
        torch.allclose(
            target[:, :, t],
            output[t + (kernel_size - 1 - reg.tcn.pad)],
            atol=5e-7,
        )
        for t in range(reg.tcn.pad, T - (kernel_size - 1))
    ]

    assert all(checks)


def test_CoStGcnBlock_residual_eq_channels():
    in_channels = 4
    out_channels = 4
    A = ntu_rgbd.graph.A
    stride = 1
    residual = True

    reg = StGcnBlock(in_channels, out_channels, A, stride, residual)
    co = CoStGcnBlock(in_channels, out_channels, A, stride, residual)

    #  Transfer weights
    state_dict = reg.state_dict()
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
        output.append(co(sample[:, :, i]))

    checks = [
        torch.allclose(
            target[:, :, t],
            output[t + (kernel_size - 1 - reg.tcn.pad)],
            atol=5e-7,
        )
        for t in range(reg.tcn.pad, T - (kernel_size - 1))
    ]

    assert all(checks)


def test_CoStGcnBlock_residual_neq_channels():
    in_channels = 2
    out_channels = 4
    A = ntu_rgbd.graph.A
    stride = 1
    residual = True

    reg = StGcnBlock(in_channels, out_channels, A, stride, residual)
    co = CoStGcnBlock(in_channels, out_channels, A, stride, residual)

    #  Transfer weights
    state_dict = reg.state_dict()
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
        output.append(co(sample[:, :, i]))

    checks = [
        torch.allclose(
            target[:, :, t],
            output[t + (kernel_size - 1 - reg.tcn.pad)],
            atol=5e-7,
        )
        for t in range(reg.tcn.pad, T - (kernel_size - 1))
    ]

    assert all(checks)


def test_CoStGcnBlock_residual_neq_channels_strided():
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

    reg = StGcnBlock(in_channels, out_channels, A, stride, residual)
    co = CoStGcnBlock(in_channels, out_channels, A, stride, residual)

    #  Transfer weights
    state_dict = reg.state_dict()
    co.load_state_dict(state_dict)

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    # Forward
    target = reg(sample)

    # Frame-by-frame
    output = []
    for i in range(T):
        output.append(co(sample[:, :, i]))

    checks = [
        torch.allclose(
            target[:, :, t],
            output[t * 2 + co.delay],
            atol=5e-7,
        )
        for t in range(reg.tcn.pad // stride, (T - co.tcn.pad) // stride)
    ]

    assert all(checks)


def test_simple_stgcn():
    # Prepare data
    T = 40
    B = 2
    in_channels = 2
    out_channels = 4
    num_nodes = ntu_rgbd.graph.A.shape[-1]
    A = ntu_rgbd.graph.A
    sample = torch.rand((B, in_channels, T, num_nodes))

    stride = 2

    reg = torch.nn.Sequential(
        StGcnBlock(in_channels, in_channels, A, residual=False),
        StGcnBlock(in_channels, in_channels, A),
        StGcnBlock(in_channels, out_channels, A, stride=stride),
    )
    co = torch.nn.Sequential(
        CoStGcnBlock(in_channels, in_channels, A, residual=False),
        CoStGcnBlock(in_channels, in_channels, A),
        CoStGcnBlock(in_channels, out_channels, A, stride=stride),
    )
    total_stride = np.prod([block.stride for block in co])
    co_delay = sum([block.delay for block in co if hasattr(block, "delay")])
    co_pads = (
        sum([block.tcn.pad for block in co if hasattr(block, "tcn")]) // total_stride
    )

    #  Transfer weights
    state_dict = reg.state_dict()
    co.load_state_dict(state_dict)

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    # Forward
    target = reg(sample)

    # Frame-by-frame
    output = []
    for i in range(T):
        output.append(co(sample[:, :, i]))

    checks = [
        torch.allclose(
            target[:, :, t],
            output[t * stride + co_delay],
            atol=5e-7,
        )
        for t in range(co_pads, (T - co_delay) // stride)
    ]

    assert all(checks)
