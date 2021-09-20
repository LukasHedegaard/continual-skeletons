from torch import nn

import continual as co
from datasets import datasets
from models.a_gcn.a_gcn import AdaptiveGraphConvolution
from models.base import CoModelBase, CoSpatioTemporalBlock
from models.utils import init_weights
from collections import OrderedDict

import ride  # isort:skip


class CoAGcn(
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
        num_classes = self.num_classes

        A = self.graph.A

        # Define layers
        data_bn = nn.BatchNorm2d(S * C_in * V)
        # Pass in precise window-sizes to compensate propperly in BatchNorm modules
        # fmt: off
        # NB: The AdaptiveGraphConvolution doesn't transfer well to continual networks since the node attention is across all timesteps
        CoGraphConv = co.forward_stepping(AdaptiveGraphConvolution)
        layers = co.Sequential(OrderedDict([
            ("layer1", CoSpatioTemporalBlock(C_in, 64, A, CoGraphConv=CoGraphConv, padding="equal", window_size=T, residual=False)),
            ("layer2", CoSpatioTemporalBlock(64, 64, A, CoGraphConv=CoGraphConv, padding="equal", window_size=T - 1 * 8)),
            ("layer3", CoSpatioTemporalBlock(64, 64, A, CoGraphConv=CoGraphConv, padding="equal", window_size=T - 2 * 8)),
            ("layer4", CoSpatioTemporalBlock(64, 64, A, CoGraphConv=CoGraphConv, padding="equal", window_size=T - 3 * 8)),
            ("layer5", CoSpatioTemporalBlock(64, 128, A, CoGraphConv=CoGraphConv, padding="equal", window_size=T - 4 * 8, stride=2)),
            ("layer6", CoSpatioTemporalBlock(128, 128, A, CoGraphConv=CoGraphConv, padding="equal", window_size=(T - 4 * 8) / 2 - 1 * 8)),
            ("layer7", CoSpatioTemporalBlock(128, 128, A, CoGraphConv=CoGraphConv, padding="equal", window_size=(T - 4 * 8) / 2 - 2 * 8)),
            ("layer8", CoSpatioTemporalBlock(128, 256, A, CoGraphConv=CoGraphConv, padding="equal", window_size=(T - 4 * 8) / 2 - 3 * 8, stride=2)),
            ("layer9", CoSpatioTemporalBlock(256, 256, A, CoGraphConv=CoGraphConv, padding="equal", window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 1 * 8)),
            ("layer10", CoSpatioTemporalBlock(256, 256, A, CoGraphConv=CoGraphConv, padding="equal", window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 2 * 8)),
        ]))
        # fmt: on
        pool_size = hparams.pool_size
        if pool_size == -1:
            pool_size = T - layers.receptive_field

        pool = co.AvgPool1d(pool_size)
        fc = nn.Linear(256, num_classes)

        # Initialize weights
        init_weights(data_bn, bs=1)
        init_weights(fc, bs=num_classes)

        # Add blocks sequentially
        co.Sequential.__init__(
            self,
            OrderedDict(
                [
                    ("data_bn", data_bn),
                    ("layers", layers),
                    ("pool", pool),
                    ("fc", fc),
                ]
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    ride.Main(CoAGcn).argparse()
