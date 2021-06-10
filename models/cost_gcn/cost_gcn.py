"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""
import ride  # isort:skip
import torch.nn as nn

from continual import AvgPoolCo1d
from datasets import datasets
from models.base import CoStGcnBase, CoStGcnBlock
from models.utils import calc_momentum, init_weights


class CoStGcn(
    ride.RideModule,
    ride.TopKAccuracyMetric(1),
    ride.optimizers.SgdOneCycleOptimizer,
    datasets.GraphDatasets,
    CoStGcnBase,
):
    def __init__(self, hparams):
        # Shapes from Dataset:
        # num_channels, num_frames, num_vertices, num_skeletons
        (C_in, T, V, S) = self.input_shape
        num_classes = self.num_classes

        A = self.graph.A

        # BN momentum should match that of clip-based inference
        # The number of frames is reduced as stride increases
        bn_mom1 = calc_momentum(T)
        bn_mom2 = calc_momentum(T // 2)
        bn_mom3 = calc_momentum(T // (2 * 2))

        # Define layers
        # fmt: off
        self.data_bn = nn.BatchNorm1d(S * C_in * V, momentum=bn_mom1)
        self.layers = nn.ModuleDict(
            {
                "layer1": CoStGcnBlock(C_in, 64, A, bn_momentum=bn_mom1, t_padding=4, residual=False),
                "layer2": CoStGcnBlock(64, 64, A, bn_momentum=bn_mom1, t_padding=4),
                "layer3": CoStGcnBlock(64, 64, A, bn_momentum=bn_mom1, t_padding=4),
                "layer4": CoStGcnBlock(64, 64, A, bn_momentum=bn_mom1, t_padding=4),
                "layer5": CoStGcnBlock(64, 128, A, bn_momentum=bn_mom2, t_padding=4, stride=2),
                "layer6": CoStGcnBlock(128, 128, A, bn_momentum=bn_mom2, t_padding=4),
                "layer7": CoStGcnBlock(128, 128, A, bn_momentum=bn_mom2, t_padding=4),
                "layer8": CoStGcnBlock(128, 256, A, bn_momentum=bn_mom3, t_padding=4, stride=2),
                "layer9": CoStGcnBlock(256, 256, A, bn_momentum=bn_mom3, t_padding=4),
                "layer10": CoStGcnBlock(256, 256, A, bn_momentum=bn_mom3, t_padding=4),
            }
        )
        # fmt: on
        if self.hparams.pool_size == -1:
            self.hparams.pool_size = T // self.stride - self.delay_conv_blocks
        self.pool = AvgPoolCo1d(window_size=self.hparams.pool_size)
        self.fc = nn.Linear(256, num_classes)

        # Initialize weights
        init_weights(self.data_bn, bs=1)
        init_weights(self.fc, bs=num_classes)

        # Defined in CoStGcnBase:
        # - self.stride
        # - self.delay_conv_blocks
        # - self.forward


if __name__ == "__main__":  # pragma: no cover
    ride.Main(CoStGcn).argparse()
