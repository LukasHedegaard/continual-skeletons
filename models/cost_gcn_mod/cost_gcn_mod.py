"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""
import ride  # isort:skip
import torch.nn as nn

from continual import AvgPoolCo1d
from datasets import datasets
from models.base import CoStGcnBase, CoStGcnBlock
from models.utils import calc_momentum, init_weights


class CoStGcnMod(
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

        # BN momentum should match that of clip-based inference
        mom = calc_momentum(num_frames)

        # Define layers
        self.data_bn = nn.BatchNorm1d(
            num_skeletons * num_channels * num_vertices, momentum=mom
        )
        self.layers = nn.ModuleDict(
            {
                "layer1": CoStGcnBlock(
                    num_channels, 64, A, residual=False, bn_momentum=mom
                ),
                "layer2": CoStGcnBlock(64, 64, A, bn_momentum=mom),
                "layer3": CoStGcnBlock(64, 64, A, bn_momentum=mom),
                "layer4": CoStGcnBlock(64, 64, A, bn_momentum=mom),
                "layer5": CoStGcnBlock(64, 128, A, bn_momentum=mom, stride=1),
                "layer6": CoStGcnBlock(128, 128, A, bn_momentum=mom),
                "layer7": CoStGcnBlock(128, 128, A, bn_momentum=mom),
                "layer8": CoStGcnBlock(128, 256, A, bn_momentum=mom, stride=1),
                "layer9": CoStGcnBlock(256, 256, A, bn_momentum=mom),
                "layer10": CoStGcnBlock(256, 256, A, bn_momentum=mom),
            }
        )
        self.pool_size = hparams.pool_size
        if self.pool_size == -1:
            self.pool_size = (
                num_frames // self.stride
                - self.delay_conv_blocks
                - 10 * 8  # No padding was used in ST-GCN-mod, hence the layers shrink
            )
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
