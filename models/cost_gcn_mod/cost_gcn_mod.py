"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""
import ride  # isort:skip

import numpy as np
import torch.nn as nn

from datasets import datasets
from models.utils import init_weights
from models.cost_gcn.cost_gcn import CoStGcnBlock, calc_momentum
from continual import (
    AdaptiveAvgPoolCo2d,
    TensorPlaceholder,
)


class CoStGcnMod(
    ride.RideModule,
    ride.TopKAccuracyMetric(1),
    ride.optimizers.SgdOptimizer,
    datasets.GraphDatasets,
):
    @staticmethod
    def configs() -> ride.Configs:
        c = ride.Configs()
        c.add(
            name="forward_mode",
            type=str,
            default="clip",
            choices=["clip", "frame"],
            strategy="choice",
            description="Run forward on a whole clip or a single frame.",
        )
        c.add(
            name="predict_after_frames",
            type=int,
            default=0,
            strategy="choice",
            description="Predict the final results after N frames.",
        )
        c.add(
            name="continual_temporal_fill",
            type=str,
            default="replicate",
            choices=["zeros", "replicate"],
            strategy="choice",
            description="Fill mode for samples along temporal dimension.",
        )
        c.add(
            name="graph",
            type=str,
            default="ntu_rgbd",
            choices=["ntu_rgbd", "kinetics"],
        )
        c.add(
            name="pool_size",
            type=int,
            default=-1,
            description=(
                "Size of pooling layer. If `-1`, a pool size will be chosen, "
                "which matches the network delay to the input_shape"
            ),
        )
        return c

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
                - self.delay_stgcn_blocks
                - 10 * 8  # No padding was used in ST-GCN-mod, hence the layers shrink
            )
        self.pool = AdaptiveAvgPoolCo2d(window_size=self.pool_size, output_size=(1,))
        self.fc = nn.Linear(256, num_classes)

        # Initialize weights
        init_weights(self.data_bn, bs=1)
        init_weights(self.fc, bs=num_classes)

        if getattr(self.hparams, "profile_model", False):
            # A new output is created every `self.stride` frames.
            self.input_shape = (num_channels, self.stride, num_vertices, num_skeletons)

    def forward(self, x):
        result = None
        if self.hparams.forward_mode == "clip":
            result = self.forward_clip(x)
        elif self.hparams.forward_mode == "frame":
            result = self.forward_frame(x[:, :, 0])
        else:
            raise RuntimeError("Model forward_mode should be one of {'clip', 'frame'}.")
        return result

    def forward_clip(self, x):
        """Forward clip.
        Initialise network with first frames and predict on the last
        """
        self.clean_states()
        result = None
        N, C, T, V, M = x.shape
        for t in range(self.hparams.predict_after_frames or T):
            result = self.forward_frame(x[:, :, t])
        return result

    def forward_frame(self, x):
        self.clean_states_on_shape_change(x.shape)
        N, C, V, M = x.shape
        x = x.permute(0, 3, 2, 1).contiguous().view(N, M * V * C, 1)
        x = self.data_bn(x)
        x = x.view(N, M, V, C).permute(0, 1, 3, 2).contiguous().view(N * M, C, V)
        for i in range(len(self.layers)):
            x = self.layers[f"layer{i + 1}"](x)
            if type(x) is TensorPlaceholder:
                # if x is None:
                return self.last_output
        # N*M,C,V
        C_new = x.size(1)
        x = self.pool(x)
        x = x.view(N, M, C_new).mean(1)
        x = self.fc(x)
        self.last_output = x
        return x

    def clean_states(self):
        for m in self.modules():
            if hasattr(m, "clean_state"):
                m.clean_state()

    def clean_states_on_shape_change(self, shape):
        if not hasattr(self, "_current_input_shape"):
            self._current_input_shape = shape

        if self._current_input_shape != shape:
            self.clean_states()

    @property
    def delay(self):
        return self.delay_stgcn_blocks + self.pool.delay

    @property
    def delay_stgcn_blocks(self):
        d = 0
        for i in range(len(self.layers)):
            d += self.layers[f"layer{i + 1}"].delay
            d = d // self.layers[f"layer{i + 1}"].stride
        return d

    @property
    def stride(self):
        s = int(
            np.prod(
                [self.layers[f"layer{i + 1}"].stride for i in range(len(self.layers))]
            )
        )
        return s


if __name__ == "__main__":  # pragma: no cover
    ride.Main(CoStGcnMod).argparse()
