"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""
import ride  # isort:skip

from collections import OrderedDict

import continual as co
from datasets import GraphDatasets
from models.base import CoModelBase, CoSpatioTemporalBlock


class CoStGcnMod(
    ride.RideModule,
    ride.TopKAccuracyMetric(1, 3, 5),
    ride.optimizers.SgdOneCycleOptimizer,
    GraphDatasets,
    CoModelBase,
):
    def __init__(self, hparams):
        # Shapes from Dataset:
        # num_channels, num_frames, num_vertices, num_skeletons
        (C_in, T, _, _) = self.input_shape
        A = self.graph.A

        # Pass in precise window-sizes to compensate propperly in BatchNorm modules
        # fmt: off
        self.layers = co.Sequential(OrderedDict([
            ("layer1", CoSpatioTemporalBlock(C_in, 64, A, padding=0, window_size=T, residual=False)),
            ("layer2", CoSpatioTemporalBlock(64, 64, A, padding=0, window_size=T - 1 * 8)),
            ("layer3", CoSpatioTemporalBlock(64, 64, A, padding=0, window_size=T - 2 * 8)),
            ("layer4", CoSpatioTemporalBlock(64, 64, A, padding=0, window_size=T - 3 * 8)),
            ("layer5", CoSpatioTemporalBlock(64, 128, A, padding=0, window_size=T - 4 * 8, stride=1)),
            ("layer6", CoSpatioTemporalBlock(128, 128, A, padding=0, window_size=(T - 4 * 8) / 2 - 1 * 8)),
            ("layer7", CoSpatioTemporalBlock(128, 128, A, padding=0, window_size=(T - 4 * 8) / 2 - 2 * 8)),
            ("layer8", CoSpatioTemporalBlock(128, 256, A, padding=0, window_size=(T - 4 * 8) / 2 - 3 * 8, stride=1)),
            ("layer9", CoSpatioTemporalBlock(256, 256, A, padding=0, window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 1 * 8)),
            ("layer10", CoSpatioTemporalBlock(256, 256, A, padding=0, window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 2 * 8)),
        ]))
        # fmt: on

        # Other layers defined in CoModelBase.on_init_end


if __name__ == "__main__":  # pragma: no cover
    ride.Main(CoStGcnMod).argparse()
