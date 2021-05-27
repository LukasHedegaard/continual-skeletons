import ride  # isort:skip

import numpy as np
import torch
import torch.nn as nn

from datasets import datasets
from models.utils import init_weights


class AGcn(
    ride.RideModule,
    ride.TopKAccuracyMetric(1),
    ride.SgdOneCycleOptimizer,
    ride.finetune.Finetunable,
    datasets.GraphDatasets,
):
    @staticmethod
    def configs() -> ride.Configs:
        c = ride.Configs()
        c.add(
            name="graph",
            type=str,
            default="ntu_rgbd",
            choices=["ntu_rgbd", "kinetics"],
        )
        return c

    def __init__(self, hparams):
        # Shapes from Dataset:
        (
            num_channels,
            num_frames,
            num_vertices,
            num_skeletons,
        ) = self.input_shape
        num_classes = self.num_classes
        A = self.graph.A

        # Define layers
        self.data_bn = nn.BatchNorm1d(num_skeletons * num_channels * num_vertices)
        self.layers = nn.ModuleDict(
            {
                "layer1": AdaptiveGcnBlock(num_channels, 64, A, residual=False),
                "layer2": AdaptiveGcnBlock(64, 64, A),
                "layer3": AdaptiveGcnBlock(64, 64, A),
                "layer4": AdaptiveGcnBlock(64, 64, A),
                "layer5": AdaptiveGcnBlock(64, 128, A, stride=2),
                "layer6": AdaptiveGcnBlock(128, 128, A),
                "layer7": AdaptiveGcnBlock(128, 128, A),
                "layer8": AdaptiveGcnBlock(128, 256, A, stride=2),
                "layer9": AdaptiveGcnBlock(256, 256, A),
                "layer10": AdaptiveGcnBlock(256, 256, A),
            }
        )
        self.fc = nn.Linear(256, num_classes)

        # Initialize weights
        init_weights(self.data_bn, bs=1)
        init_weights(self.fc, bs=num_classes)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = (
            x.view(N, M, V, C, T)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, T, V)
        )
        for i in range(len(self.layers)):
            x = self.layers[f"layer{i + 1}"](x)
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.fc(x)
        return x


class AdaptiveGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4):
        super(AdaptiveGraphConvolution, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.graph_attn = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.graph_attn, 1)
        self.A = nn.Parameter(
            torch.from_numpy(A.astype(np.float32)), requires_grad=False
        )
        self.num_subset = 3
        self.g_conv = nn.ModuleList()
        self.a_conv = nn.ModuleList()
        self.b_conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.g_conv.append(nn.Conv2d(in_channels, out_channels, 1))
            self.a_conv.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.b_conv.append(nn.Conv2d(in_channels, inter_channels, 1))
            init_weights(self.g_conv[i], bs=self.num_subset)
            init_weights(self.a_conv[i])
            init_weights(self.b_conv[i])

        if in_channels != out_channels:
            self.gcn_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
            init_weights(self.gcn_residual[0], bs=1)
            init_weights(self.gcn_residual[1], bs=1)
        else:
            self.gcn_residual = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        init_weights(self.bn, bs=1e-6)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(-2)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A + self.graph_attn
        hidden_ = None
        for i in range(self.num_subset):
            A1 = (
                self.a_conv[i](x)
                .permute(0, 3, 1, 2)
                .contiguous()
                .view(N, V, self.inter_c * T)
            )
            A2 = self.b_conv[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            x_a = x.view(N, C * T, V)
            z = self.g_conv[i](torch.matmul(x_a, A1).view(N, C, T, V))
            hidden_ = z + hidden_ if hidden_ is not None else z
        hidden_ = self.bn(hidden_)
        hidden_ += self.gcn_residual(x)
        return self.relu(hidden_)


class TemporalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TemporalConvolution, self).__init__()

        self.pad = int((kernel_size - 1) / 2)
        self.t_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(self.pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        init_weights(self.t_conv, bs=1)
        init_weights(self.bn, bs=1)

    def forward(self, x):
        x = self.bn(self.t_conv(x))
        return x


def zero(x):
    return 0


def unity(x):
    return x


class AdaptiveGcnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(AdaptiveGcnBlock, self).__init__()

        self.gcn = AdaptiveGraphConvolution(in_channels, out_channels, A)
        self.tcn = TemporalConvolution(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = zero
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = unity
        else:
            self.residual = TemporalConvolution(
                in_channels, out_channels, kernel_size=1, stride=stride
            )

    def forward(self, x):
        z = self.tcn(self.gcn(x))
        r = self.residual(x)
        return self.relu(z + r)


if __name__ == "__main__":  # pragma: no cover
    ride.Main(AGcn).argparse()
