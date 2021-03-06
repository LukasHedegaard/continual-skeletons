"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""
import ride  # isort:skip

import torch.nn as nn

from datasets import datasets
from models.base import SpatioTemporalBlock
from models.utils import init_weights


class StGcnMod(
    ride.RideModule,
    ride.TopKAccuracyMetric(1, 3, 5),
    ride.optimizers.SgdOneCycleOptimizer,
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
        self.layers = nn.ModuleDict(
            {
                "layer1": SpatioTemporalBlock(
                    num_channels, 64, A, residual=False, temporal_padding=0
                ),
                "layer2": SpatioTemporalBlock(64, 64, A, temporal_padding=0),
                "layer3": SpatioTemporalBlock(64, 64, A, temporal_padding=0),
                "layer4": SpatioTemporalBlock(64, 64, A, temporal_padding=0),
                "layer5": SpatioTemporalBlock(64, 128, A, temporal_padding=0, stride=1),
                "layer6": SpatioTemporalBlock(128, 128, A, temporal_padding=0),
                "layer7": SpatioTemporalBlock(128, 128, A, temporal_padding=0),
                "layer8": SpatioTemporalBlock(
                    128, 256, A, temporal_padding=0, stride=1
                ),
                "layer9": SpatioTemporalBlock(256, 256, A, temporal_padding=0),
                "layer10": SpatioTemporalBlock(256, 256, A, temporal_padding=0),
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


if __name__ == "__main__":  # pragma: no cover
    ride.Main(StGcnMod).argparse()
