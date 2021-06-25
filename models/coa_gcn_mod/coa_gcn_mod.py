import ride  # isort:skip
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from continual import AvgPoolCo1d, BatchNormCo2d
from continual.batchnorm import normalise_momentum
from continual.utils import temporary_parameter
from datasets import datasets
from models.base import CoStGcnBase, CoStGcnBlock
from models.utils import init_weights


class CoAdaptiveGraphConvolution(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        bn_momentum=0.1,
        coff_embedding=4,
        window_size=1,
    ):
        super(CoAdaptiveGraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unnormalised_momentum = bn_momentum
        self.momentum = normalise_momentum(bn_momentum, window_size)
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
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=self.momentum),
            )
            init_weights(self.gcn_residual[0], bs=1)
            init_weights(self.gcn_residual[1], bs=1)
        else:
            self.gcn_residual = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels, momentum=self.momentum)
        init_weights(self.bn, bs=1e-6)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(-2)

    def forward(self, x):
        x = x.unsqueeze(dim=2)
        x = self._forward(x)
        x = x.squeeze(dim=2)
        return x

    def forward_regular(self, x: Tensor) -> Tensor:
        return self.forward_regular_unrolled(x)

    def forward_regular_unrolled(self, x: Tensor) -> Tensor:
        with temporary_parameter(
            self.bn, "momentum", self.unnormalised_momentum
        ), temporary_parameter(
            self.gcn_residual[-1], "momentum", self.unnormalised_momentum
        ) if self.in_channels != self.out_channels else nullcontext():
            x = self._forward(x)
        return x

    def _forward(self, x: Tensor) -> Tensor:
        N, C, T, V = x.size()
        A = self.A + self.graph_attn
        hidden = None
        for i in range(self.num_subset):
            A1 = self.a_conv[i](x).permute(0, 3, 1, 2).contiguous()
            A2 = self.b_conv[i](x)
            A1 = self.soft(
                torch.einsum("nvct,nctw->nvwt", A1, A2) / self.inter_c
            )  # N V V T
            A1 = A1 + A[i].unsqueeze(0).unsqueeze(-1)
            z = self.g_conv[i](torch.einsum("nctv,nvwt->nctv", x, A1))
            hidden = z + hidden if hidden is not None else z
        hidden = self.bn(hidden)
        hidden += self.gcn_residual(x)
        return self.relu(hidden)


class CoAGcnMod(
    ride.RideModule,
    ride.TopKAccuracyMetric(1),
    ride.optimizers.SgdOptimizer,
    datasets.GraphDatasets,
    CoStGcnBase,
):
    def __init__(self, hparams):
        # Shapes from Dataset:
        (num_channels, num_frames, num_vertices, num_skeletons) = self.input_shape
        num_classes = self.num_classes

        A = self.graph.A

        # Define layers
        self.data_bn = BatchNormCo2d(
            num_skeletons * num_channels * num_vertices, window_size=num_frames
        )
        # Pass in precise window-sizes to compensate propperly in BatchNorm modules
        # fmt: off
        self.layers = nn.ModuleDict(
            {
                "layer1": CoStGcnBlock(num_channels, 64, A, CoGraphConv=CoAdaptiveGraphConvolution, padding=0, window_size=num_frames, residual=False),
                "layer2": CoStGcnBlock(64, 64, A, CoGraphConv=CoAdaptiveGraphConvolution, padding=0, window_size=num_frames - 1 * 8),
                "layer3": CoStGcnBlock(64, 64, A, CoGraphConv=CoAdaptiveGraphConvolution, padding=0, window_size=num_frames - 2 * 8),
                "layer4": CoStGcnBlock(64, 64, A, CoGraphConv=CoAdaptiveGraphConvolution, padding=0, window_size=num_frames - 3 * 8),
                "layer5": CoStGcnBlock(64, 128, A, CoGraphConv=CoAdaptiveGraphConvolution, padding=0, window_size=num_frames - 4 * 8, stride=1),
                "layer6": CoStGcnBlock(128, 128, A, CoGraphConv=CoAdaptiveGraphConvolution, padding=0, window_size=(num_frames - 4 * 8) / 2 - 1 * 8),
                "layer7": CoStGcnBlock(128, 128, A, CoGraphConv=CoAdaptiveGraphConvolution, padding=0, window_size=(num_frames - 4 * 8) / 2 - 2 * 8),
                "layer8": CoStGcnBlock(128, 256, A, CoGraphConv=CoAdaptiveGraphConvolution, padding=0, window_size=(num_frames - 4 * 8) / 2 - 3 * 8, stride=1),
                "layer9": CoStGcnBlock(256, 256, A, CoGraphConv=CoAdaptiveGraphConvolution, padding=0, window_size=((num_frames - 4 * 8) / 2 - 3 * 8) / 2 - 1 * 8),
                "layer10": CoStGcnBlock(256, 256, A, CoGraphConv=CoAdaptiveGraphConvolution, padding=0, window_size=((num_frames - 4 * 8) / 2 - 3 * 8) / 2 - 2 * 8),
            }
        )
        # fmt: on
        self.pool_size = hparams.pool_size
        if self.pool_size == -1:
            self.pool_size = num_frames // self.stride - self.delay_conv_blocks
        self.pool = AvgPoolCo1d(window_size=self.pool_size)
        self.fc = nn.Linear(256, num_classes)

        # Initialize weights
        init_weights(self.data_bn, bs=1)
        init_weights(self.fc, bs=num_classes)

        # Defined in CoStGcnBase:
        # - self.stride
        # - self.delay_conv_blocks
        # - self.forward


if __name__ == "__main__":  # pragma: no cover
    ride.Main(CoAGcnMod).argparse()
