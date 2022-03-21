import ride  # isort:skip

import numpy as np
import torch
import torch.nn as nn

from datasets import datasets
from models.base import SpatioTemporalBlock
from models.utils import init_weights


class AdaptiveGraphConvolutionMod(nn.Module):
    def __init__(self, in_channels, out_channels, A, bn_momentum=0.1, coff_embedding=4):
        super(AdaptiveGraphConvolutionMod, self).__init__()
        self.inter_c = out_channels // coff_embedding
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
            self.a_conv.append(nn.Conv2d(in_channels, self.inter_c, 1))
            self.b_conv.append(nn.Conv2d(in_channels, self.inter_c, 1))
            init_weights(self.g_conv[i], bs=self.num_subset)
            init_weights(self.a_conv[i])
            init_weights(self.b_conv[i])

        if in_channels != out_channels:
            self.gcn_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            )
            init_weights(self.gcn_residual[0], bs=1)
            init_weights(self.gcn_residual[1], bs=1)
        else:
            self.gcn_residual = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        init_weights(self.bn, bs=1e-6)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(-2)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A + self.graph_attn
        hidden = None
        for i in range(self.num_subset):
            A1 = self.a_conv[i](x).permute(0, 3, 1, 2).contiguous()
            A2 = self.b_conv[i](x)
            # Modified attention within timestep
            A1 = self.soft(
                torch.einsum("nvct,nctw->nvwt", A1, A2) / self.inter_c
            )  # N V V T
            A1 = A1 + A[i].unsqueeze(0).unsqueeze(-1)
            z = self.g_conv[i](torch.einsum("nctv,nvwt->nctv", x, A1))
            hidden = z + hidden if hidden is not None else z
        hidden = self.bn(hidden)
        hidden += self.gcn_residual(x)
        return self.relu(hidden)


class AGcnMod(
    ride.RideModule,
    ride.TopKAccuracyMetric(1, 3, 5),
    ride.SgdOneCycleOptimizer,
    ride.finetune.Finetunable,
    datasets.GraphDatasets,
):
    def __init__(self, hparams):
        # Shapes from Dataset:
        (num_channels, num_frames, num_vertices, num_skeletons) = self.input_shape
        num_classes = self.num_classes

        A = self.graph.A

        # Define layers
        self.data_bn = nn.BatchNorm1d(num_skeletons * num_channels * num_vertices)

        GraphConv = AdaptiveGraphConvolutionMod
        # fmt: off
        self.layers = nn.ModuleDict(
            {
                "layer1": SpatioTemporalBlock(num_channels, 64, A, GraphConv=GraphConv, residual=False, temporal_padding=0),
                "layer2": SpatioTemporalBlock(64, 64, A, GraphConv=GraphConv, temporal_padding=0),
                "layer3": SpatioTemporalBlock(64, 64, A, GraphConv=GraphConv, temporal_padding=0),
                "layer4": SpatioTemporalBlock(64, 64, A, GraphConv=GraphConv, temporal_padding=0),
                "layer5": SpatioTemporalBlock(64, 128, A, GraphConv=GraphConv, temporal_padding=0, stride=1),
                "layer6": SpatioTemporalBlock(128, 128, A, GraphConv=GraphConv, temporal_padding=0),
                "layer7": SpatioTemporalBlock(128, 128, A, GraphConv=GraphConv, temporal_padding=0),
                "layer8": SpatioTemporalBlock(128, 256, A, GraphConv=GraphConv, temporal_padding=0, stride=1),
                "layer9": SpatioTemporalBlock(256, 256, A, GraphConv=GraphConv, temporal_padding=0),
                "layer10": SpatioTemporalBlock(256, 256, A, GraphConv=GraphConv, temporal_padding=0),
            }
        )
        # fmt: on
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


if __name__ == "__main__":  # pragma: no cover
    ride.Main(AGcnMod).argparse()
