import numpy as np
import torch

from datasets import ntu_rgbd
from models.base import (
    CoStGcnBlock,
    CoTemporalConvolution,
    StGcnBlock,
    TemporalConvolution,
)
from models.cost_gcn.cost_gcn import CoStGcn
from models.st_gcn.st_gcn import StGcn


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
            output[t + (kernel_size - 1 - reg.padding)],
            atol=5e-7,
        )
        for t in range(reg.padding, sample.shape[2] - (kernel_size - 1))
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
            output[t + (kernel_size - 1 - reg.tcn.padding)],
            atol=5e-7,
        )
        for t in range(reg.tcn.padding, T - (kernel_size - 1))
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
            output[t + (kernel_size - 1 - reg.tcn.padding)],
            atol=5e-7,
        )
        for t in range(reg.tcn.padding, T - (kernel_size - 1))
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
            output[t + (kernel_size - 1 - reg.tcn.padding)],
            atol=5e-7,
        )
        for t in range(reg.tcn.padding, T - (kernel_size - 1))
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
            output[t * 2 + co.delay // 2],
            atol=5e-7,
        )
        for t in range(reg.tcn.padding // stride, (T - co.tcn.delay // 2) // stride)
    ]

    assert all(checks)


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
    wait = sum([block.tcn.kernel_size // 2 for block in co if hasattr(block, "tcn")])

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
            output[t * total_stride + wait],
            atol=5e-6,
        )
        for t in range(wait // 2, (T - wait) // total_stride)
    ]

    assert all(checks)


def test_costgcn_until_pool():
    # Model definition
    hparams = default_hparams()
    hparams["dataset_name"] = "dummy"
    reg = StGcn(hparams)
    co = CoStGcn(hparams)

    #  Transfer weights
    co.load_state_dict(reg.state_dict())

    # Set to eval mode (otherwise batchnorm doesn't match)
    reg.eval()
    co.eval()

    # Register forward hook
    reg_features = []
    co_features = []

    def reg_store_features(sself, input, output):
        nonlocal reg_features
        reg_features = output

    def co_store_features(sself, input, output):
        nonlocal co_features
        co_features.append(output)

    reg.layers.layer10.register_forward_hook(reg_store_features)
    co.layers.layer10.register_forward_hook(co_store_features)

    # Prepare sample
    sample = torch.randn(reg.hparams.batch_size, *reg.input_shape)

    # Forward
    reg(sample)
    co(sample)

    # (4 * 5 / 2 + 3 * 4) / 2 + 2 * 4 = 19
    delay = 0
    for i in range(len(co.layers)):
        delay += co.layers[f"layer{i + 1}"].delay
        delay = delay // co.layers[f"layer{i + 1}"].stride

    # Match for reg =----=
    #           co  ==----
    # where '=' is delay
    checks = [
        torch.allclose(reg_features[:, :, i], co_features[i + delay], atol=5e-5)
        for i in range(delay, len(co_features) - delay)
    ]

    assert all(checks)
