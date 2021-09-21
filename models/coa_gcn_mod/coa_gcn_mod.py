import ride  # isort: skip
from typing import OrderedDict

import numpy as np
import torch
from torch import Tensor, nn

import continual as co
from datasets import datasets
from models.base import CoModelBase, CoSpatioTemporalBlock
from models.utils import init_weights


class TimeSlicedAdaptiveGraphConvolution(co.CoModule, nn.Module):
    # Alternative implementation of AdaptiveGraphConvolution, which does not collapse multiple time-steps together.
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        bn_momentum=0.1,
        coff_embedding=4,
    ):
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.momentum = bn_momentum
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

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
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
            z = self.g_conv[i](torch.einsum("nctv,nwvt->nctv", x, A1))
            hidden = z + hidden if hidden is not None else z
        hidden = self.bn(hidden)
        hidden += self.gcn_residual(x)
        return self.relu(hidden)

    def forward_step(self, x: Tensor, update_state=True) -> Tensor:
        x = x.unsqueeze(2)
        x = self.forward(x)
        x = x.squeeze(2)
        return x

    def forward_steps(self, x: Tensor, pad_end=False, update_state=True) -> Tensor:
        return self.forward(x)


class CoAGcnMod(
    ride.RideModule,
    ride.TopKAccuracyMetric(1),
    ride.optimizers.SgdOneCycleOptimizer,
    datasets.GraphDatasets,
    CoModelBase,
):
    def __init__(self, hparams):
        # Shapes from Dataset:
        # num_channels, num_frames, num_vertices, num_skeletons
        (C_in, T, V, S) = self.input_shape
        A = self.graph.A

        CoGraphConv = TimeSlicedAdaptiveGraphConvolution
        # fmt: off
        self.layers = co.Sequential(OrderedDict([
            ("layer1", CoSpatioTemporalBlock(C_in, 64, A, CoGraphConv=CoGraphConv, padding=0, window_size=T, residual=False)),
            ("layer2", CoSpatioTemporalBlock(64, 64, A, CoGraphConv=CoGraphConv, padding=0, window_size=T - 1 * 8)),
            ("layer3", CoSpatioTemporalBlock(64, 64, A, CoGraphConv=CoGraphConv, padding=0, window_size=T - 2 * 8)),
            ("layer4", CoSpatioTemporalBlock(64, 64, A, CoGraphConv=CoGraphConv, padding=0, window_size=T - 3 * 8)),
            ("layer5", CoSpatioTemporalBlock(64, 128, A, CoGraphConv=CoGraphConv, padding=0, window_size=T - 4 * 8, stride=1)),
            ("layer6", CoSpatioTemporalBlock(128, 128, A, CoGraphConv=CoGraphConv, padding=0, window_size=(T - 4 * 8) / 2 - 1 * 8)),
            ("layer7", CoSpatioTemporalBlock(128, 128, A, CoGraphConv=CoGraphConv, padding=0, window_size=(T - 4 * 8) / 2 - 2 * 8)),
            ("layer8", CoSpatioTemporalBlock(128, 256, A, CoGraphConv=CoGraphConv, padding=0, window_size=(T - 4 * 8) / 2 - 3 * 8, stride=1)),
            ("layer9", CoSpatioTemporalBlock(256, 256, A, CoGraphConv=CoGraphConv, padding=0, window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 1 * 8)),
            ("layer10", CoSpatioTemporalBlock(256, 256, A, CoGraphConv=CoGraphConv, padding=0, window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 2 * 8)),
        ]))
        # fmt: on

        # Other layers defined in CoModelBase.on_init_end


if __name__ == "__main__":  # pragma: no cover
    ride.Main(CoAGcnMod).argparse()
