import ride  # isort:skip

import torch.nn as nn

from continual import AvgPoolCo1d
from datasets import datasets
from models.a_gcn.a_gcn import AdaptiveGraphConvolution
from models.base import CoStGcnBase, CoStGcnBlock
from models.utils import init_weights


class CoAGcn(
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
        self.data_bn = nn.BatchNorm1d(num_skeletons * num_channels * num_vertices)
        GraphConv = AdaptiveGraphConvolution
        self.layers = nn.ModuleDict(
            {
                "layer1": CoStGcnBlock(
                    num_channels, 64, A, GraphConv=GraphConv, residual=False
                ),
                "layer2": CoStGcnBlock(64, 64, A, GraphConv=GraphConv),
                "layer3": CoStGcnBlock(64, 64, A, GraphConv=GraphConv),
                "layer4": CoStGcnBlock(64, 64, A, GraphConv=GraphConv),
                "layer5": CoStGcnBlock(64, 128, A, GraphConv=GraphConv, stride=2),
                "layer6": CoStGcnBlock(128, 128, A, GraphConv=GraphConv),
                "layer7": CoStGcnBlock(128, 128, A, GraphConv=GraphConv),
                "layer8": CoStGcnBlock(128, 256, A, GraphConv=GraphConv, stride=2),
                "layer9": CoStGcnBlock(256, 256, A, GraphConv=GraphConv),
                "layer10": CoStGcnBlock(256, 256, A, GraphConv=GraphConv),
            }
        )
        self.pool_size = hparams.pool_size
        if self.pool_size == -1:
            self.pool_size = num_frames // self.stride - self.delay_conv_blocks
        self.pool = AvgPoolCo1d(window_size=self.pool_size)

        self.fc = nn.Linear(256, num_classes)

        # Initialize weights
        init_weights(self.data_bn, bs=1)
        init_weights(self.fc, bs=num_classes)


if __name__ == "__main__":  # pragma: no cover
    ride.Main(CoAGcn).argparse()
