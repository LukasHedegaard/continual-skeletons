"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""
import ride  # isort:skip
import torch.nn as nn

from continual import AvgPoolCo1d, BatchNormCo2d
from datasets import datasets
from models.base import CoStGcnBase, CoStGcnBlock
from models.utils import init_weights


class CoStGcnMod(
    ride.RideModule,
    ride.TopKAccuracyMetric(1, 3, 5),
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

        # Define layers
        self.data_bn = BatchNormCo2d(S * C_in * V, window_size=T)
        # Pass in precise window-sizes to compensate propperly in BatchNorm modules
        # fmt: off
        self.layers = nn.ModuleDict(
            {
                "layer1": CoStGcnBlock(C_in, 64, A, padding=0, window_size=T, residual=False),
                "layer2": CoStGcnBlock(64, 64, A, padding=0, window_size=T - 1 * 8),
                "layer3": CoStGcnBlock(64, 64, A, padding=0, window_size=T - 2 * 8),
                "layer4": CoStGcnBlock(64, 64, A, padding=0, window_size=T - 3 * 8),
                "layer5": CoStGcnBlock(64, 128, A, padding=0, window_size=T - 4 * 8, stride=1),
                "layer6": CoStGcnBlock(128, 128, A, padding=0, window_size=(T - 4 * 8) / 2 - 1 * 8),
                "layer7": CoStGcnBlock(128, 128, A, padding=0, window_size=(T - 4 * 8) / 2 - 2 * 8),
                "layer8": CoStGcnBlock(128, 256, A, padding=0, window_size=(T - 4 * 8) / 2 - 3 * 8, stride=1),
                "layer9": CoStGcnBlock(256, 256, A, padding=0, window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 1 * 8),
                "layer10": CoStGcnBlock(256, 256, A, padding=0, window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 2 * 8),
            }
        )
        # fmt: on
        self.pool_size = hparams.pool_size
        if self.pool_size == -1:
            self.pool_size = T // self.stride - self.delay_conv_blocks
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
    ride.Main(CoStGcnMod).argparse()
