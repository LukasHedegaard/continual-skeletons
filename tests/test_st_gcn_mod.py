import torch
from models.st_gcn_mod.st_gcn_mod import StGcnBlock
from models.cost_gcn.cost_gcn import CoStGcnBlock
from datasets import ntu_rgbd


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
