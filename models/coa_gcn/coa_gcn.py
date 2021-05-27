import ride  # isort:skip

import numpy as np
import torch
import torch.nn as nn

from datasets import datasets
from models.utils import init_weights
from continual import (
    AdaptiveAvgPoolCo2d,
    ConvCo2d,
    unsqueezed,
    Delay,
    TensorPlaceholder,
)


class CoAGcn(
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

        # Define layers
        self.data_bn = nn.BatchNorm1d(num_skeletons * num_channels * num_vertices)
        self.layers = nn.ModuleDict(
            {
                "layer1": CoAdaptiveGcnBlock(num_channels, 64, A, residual=False),
                "layer2": CoAdaptiveGcnBlock(64, 64, A),
                "layer3": CoAdaptiveGcnBlock(64, 64, A),
                "layer4": CoAdaptiveGcnBlock(64, 64, A),
                "layer5": CoAdaptiveGcnBlock(64, 128, A, stride=2),
                "layer6": CoAdaptiveGcnBlock(128, 128, A),
                "layer7": CoAdaptiveGcnBlock(128, 128, A),
                "layer8": CoAdaptiveGcnBlock(128, 256, A, stride=2),
                "layer9": CoAdaptiveGcnBlock(256, 256, A),
                "layer10": CoAdaptiveGcnBlock(256, 256, A),
            }
        )
        self.pool_size = hparams.pool_size
        if self.pool_size == -1:
            self.pool_size = num_frames // self.stride - self.delay_stgcn_blocks
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


class AdaptiveGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4):
        super(AdaptiveGraphConvolution, self).__init__()
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
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
            init_weights(self.gcn_residual[0], bs=1)
            init_weights(self.gcn_residual[1], bs=1)
        else:
            self.gcn_residual = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        init_weights(self.bn, bs=1e-6)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(-2)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A + self.graph_attn
        hidden_ = None
        for i in range(self.num_subset):
            A1 = (
                self.a_conv[i](x)
                .permute(0, 3, 1, 2)
                .contiguous()
                .view(N, V, self.inter_c * T)
            )
            A2 = self.b_conv[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            x_a = x.view(N, C * T, V)
            z = self.g_conv[i](torch.matmul(x_a, A1).view(N, C, T, V))
            hidden_ = z + hidden_ if hidden_ is not None else z
        hidden_ = self.bn(hidden_)
        hidden_ += self.gcn_residual(x)
        return self.relu(hidden_)


class CoTemporalConvolution(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        extra_delay: int = None,
    ):
        super(CoTemporalConvolution, self).__init__()

        self.pad = int((kernel_size - 1) / 2)
        self.t_conv = ConvCo2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(self.pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm1d(out_channels)
        if extra_delay:
            self.extra_delay = Delay(extra_delay)
        init_weights(self.t_conv, bs=1)
        init_weights(self.bn, bs=1)

    def forward(self, x):
        x = self.t_conv(x)
        # if x is None:
        if type(x) is TensorPlaceholder:  # Support strided conv
            return x
        x = self.bn(x)
        if hasattr(self, "extra_delay"):
            x = self.extra_delay(x)
        return x

    @property
    def delay(self):
        d = self.t_conv.delay
        if hasattr(self, "extra_delay"):
            d += self.extra_delay.delay
        return d


class CoAdaptiveGcnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(CoAdaptiveGcnBlock, self).__init__()
        self.stride = stride
        self.gcn = unsqueezed(AdaptiveGraphConvolution(in_channels, out_channels, A))
        self.tcn = CoTemporalConvolution(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (self.stride == 1):
            self.residual = Delay(self.tcn.t_conv.delay, temporal_fill="zeros")
        else:
            self.residual = CoTemporalConvolution(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=self.stride,
                extra_delay=self.tcn.t_conv.delay // self.stride,
            )

    def forward(self, x):
        if type(x) != torch.Tensor:
            print("hey")
        z = self.tcn(self.gcn(x))
        r = self.residual(x)
        if type(z) is TensorPlaceholder:
            return TensorPlaceholder(z.shape)
        return self.relu(z + r)

    @property
    def delay(self):
        return self.tcn.delay


if __name__ == "__main__":  # pragma: no cover
    ride.Main(CoAGcn).argparse()
