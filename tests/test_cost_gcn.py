import torch
from torch.nn import Conv1d, AvgPool1d

from models.cost_gcn.continual import ConvCo1d, AvgPoolCo1d
# from models.cost_gcn.cost_gcn import CoStGcn


def test_ConvCo1d():
    C = 2
    T = 3
    L = 5
    sample = torch.normal(mean=torch.zeros(L * C)).reshape((1, C, L))

    # Regular
    conv = Conv1d(in_channels=C, out_channels=1, kernel_size=T, bias=True)
    target = conv(sample)

    # Continual
    co_conv = ConvCo1d.from_regular(conv, "zeros")
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_conv(sample[:, :, i]))

    # Match after delay of T - 1
    for t in range(sample.shape[2] - (T - 1)):
        assert torch.allclose(target[:, :, t], output[t + (T - 1)])

    # Whole time-series
    output = co_conv.forward_regular(sample)
    assert torch.allclose(target, output)


def test_ConvCo1d_stride():
    C = 2
    T = 3
    L = 5
    S = 2
    sample = torch.normal(mean=torch.zeros(L * C)).reshape((1, C, L))

    # Regular
    conv = Conv1d(in_channels=C, out_channels=1, kernel_size=T, bias=True, stride=S)
    target = conv(sample)

    # Continual
    co_conv = ConvCo1d.from_regular(conv, "zeros")
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_conv(sample[:, :, i]))

    # Match after delay of T - 1
    for t in range(sample.shape[2] - (T - 1)):
        if t % S == 0:
            assert torch.allclose(target[:, :, t // S], output[t + (T - 1)])
        else:
            assert output[t + (T - 1)] is None

    # Whole time-series
    output = co_conv.forward_regular(sample)
    assert torch.allclose(target, output)


def test_AvgPoolCo1d():
    C = 2
    T = 3
    L = 5
    sample = torch.normal(mean=torch.zeros(L * C)).reshape((1, C, L))

    # Regular
    pool = AvgPool1d(T, stride=1)
    target = pool(sample)

    # Continual
    co_pool = AvgPoolCo1d(T)
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_pool(sample[:, :, i]))

    # Match after delay of T - 1
    for t in range(sample.shape[2] - (T - 1)):
        assert torch.allclose(target[:, :, t], output[t + (T - 1)])

    # Whole time-series
    output = co_pool.forward_regular(sample)
    assert torch.allclose(target, output)


# def default_hparams():
#     d = CoStGcn.configs().default_values()
#     d["max_epochs"] = 1
#     d["batch_size"] = 2
#     return d


# def xtest_forward():
#     hparams = default_hparams()
#     hparams["dataset_name"] = "dummy"
#     net = CoStGcn(hparams)

#     input = torch.rand((hparams["batch_size"], *net.input_shape))
#     output = net(input)

#     assert output.shape == (hparams["batch_size"], *net.output_shape)
