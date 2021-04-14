import torch
from torch.nn import Conv1d, Conv2d, AvgPool1d
from models.cost_gcn.continual import ConvCo1d, ConvCo2d, AvgPoolCo1d


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


def test_ConvCo2d():
    C = 2
    T = 3
    S = 2
    L = 5
    H = 3
    sample = torch.normal(mean=torch.zeros(L * C * H)).reshape((1, C, L, H))

    # Regular
    conv = Conv2d(in_channels=C, out_channels=1, kernel_size=(T, S), bias=True)
    target = conv(sample)

    # Continual
    co_conv = ConvCo2d.from_regular(conv, "zeros")
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


def test_ConvCo2d_stride():
    C = 2
    T = 3
    S = 2
    L = 5
    H = 3
    stride = 2
    sample = torch.normal(mean=torch.zeros(L * C * H)).reshape((1, C, L, H))

    # Regular
    conv = Conv2d(
        in_channels=C, out_channels=1, kernel_size=(T, S), bias=True, stride=stride
    )
    target = conv(sample)

    # Continual
    co_conv = ConvCo2d.from_regular(conv, "zeros")
    output = []

    # Frame by frame
    for i in range(sample.shape[2]):
        output.append(co_conv(sample[:, :, i]))

    # Match after delay of T - 1
    for t in range(sample.shape[2] - (T - 1)):
        if t % S == 0:
            assert torch.allclose(target[:, :, t // stride], output[t + (T - 1)])
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
