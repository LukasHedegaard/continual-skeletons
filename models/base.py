from operator import attrgetter

import numpy as np
import torch
import torch.nn as nn
from ride.core import Configs, RideMixin

from continual import ConvCo2d, Delay, TensorPlaceholder, unsqueezed
from models.utils import init_weights, unity, zero


class CoStGcnBase(RideMixin):

    hparams: ...

    @staticmethod
    def configs() -> Configs:
        c = Configs()
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
            name="pool_size",
            type=int,
            default=-1,
            description=(
                "Size of pooling layer. If `-1`, a pool size will be chosen, "
                "which matches the network delay to the input_shape"
            ),
        )
        return c

    def on_init_end(self, hparams, *args, **kwargs):
        if getattr(self.hparams, "profile_model", False):
            (num_channels, num_frames, num_vertices, num_skeletons) = self.input_shape

            # A new output is created every `self.stride` frames.
            self.input_shape = (num_channels, self.stride, num_vertices, num_skeletons)

    def validate_attributes(self):
        attrgetter("parameters")(self)
        for hparam in CoStGcnBase.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

        for attribute in [
            "data_bn",
            *[f"layers.layer{i+1}" for i in range(10)],
            "pool",
            "fc",
        ]:
            assert issubclass(type(attrgetter(attribute)(self)), torch.nn.Module)

    def forward(self, x):
        result = None
        if self.hparams.forward_mode == "clip" and self.training:
            result = self.forward_clip_efficient(x)
        elif self.hparams.forward_mode == "clip" and not self.training:
            result = self.forward_clip(x)
        elif self.hparams.forward_mode == "frame":
            result = self.forward_frame(x[:, :, 0])
        else:  # pragma: no cover
            raise RuntimeError("Model forward_mode should be one of {'clip', 'frame'}.")
        return result

    def forward_clip(self, x):
        """Forward clip.
        Initialise network with first frames and predict on the last.
        NB: This function should not be used for training.
            Because running statistics in the batch norm modules include all inputs,
            the transient response would add to these statistics.
            For training, use `forward_clip_efficient`.
        """
        self.clean_states()
        result = None
        N, C, T, V, M = x.shape
        for t in range(self.hparams.predict_after_frames or T):
            result = self.forward_frame(x[:, :, t])
        return result

    def forward_clip_efficient(self, x):
        """Efficient version of forward clip.
        Compute features layer by layer as in regular convolution.
        Produce prediction corresponding to last frame.
        """
        self.clean_states()

        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)

        x = torch.stack(
            [
                self.data_bn(x[:, :, i])
                .view(N, M, V, C)
                .permute(0, 1, 3, 2)
                .contiguous()
                .view(N * M, C, V)
                for i in range(T)
            ],
            dim=2,
        )

        for i in range(len(self.layers)):
            x = self.layers[f"layer{i + 1}"].forward_clip(x)
            # Discard frames from transient response
            discard = (
                self.layers[f"layer{i + 1}"].delay
                // self.layers[f"layer{i + 1}"].stride
            )
            x = x[:, :, discard:]

        # N*M, C, T, V
        _, C_new, T_new, _ = x.shape
        x = x.view(N, M, C_new, -1, V).mean(4).mean(1)
        assert self.pool.window_size == T_new
        x = [self.pool(x[:, :, t]) for t in range(T_new)]
        d = self.hparams.predict_after_frames - self.delay_conv_blocks
        i = d if d > 0 else -1
        x = self.fc(x[i])
        self.last_output = x
        return self.last_output

    def forward_frame(self, x):
        self.clean_states_on_shape_change(x.shape)
        N, C, V, M = x.shape
        x = x.permute(0, 3, 2, 1).contiguous().view(N, M * V * C, 1)
        x = self.data_bn(x)
        x = x.view(N, M, V, C).permute(0, 1, 3, 2).contiguous().view(N * M, C, V)
        for i in range(len(self.layers)):
            x = self.layers[f"layer{i + 1}"](x)
            if type(x) is TensorPlaceholder:
                return self.last_output
        # N*M,C,V
        C_new = x.size(1)
        x = x.view(N, M, C_new, V).mean(3).mean(1)
        x = self.pool(x)
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
        return self.delay_conv_blocks + self.pool.delay

    @property
    def delay_conv_blocks(self):
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


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A, bn_momentum=0.1, *args, **kwargs):
        super(GraphConvolution, self).__init__()
        self.graph_attn = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.graph_attn, 1)
        self.A = nn.Parameter(
            torch.from_numpy(A.astype(np.float32)), requires_grad=False
        )
        self.num_subset = 3
        self.g_conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.g_conv.append(nn.Conv2d(in_channels, out_channels, 1))
            init_weights(self.g_conv[i], bs=self.num_subset)

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

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A * self.graph_attn
        hidden_ = None
        for i in range(self.num_subset):
            x_a = x.view(N, C * T, V)
            z = self.g_conv[i](torch.matmul(x_a, A[i]).view(N, C, T, V))
            hidden_ = z + hidden_ if hidden_ is not None else z
        hidden_ = self.bn(hidden_)
        hidden_ += self.gcn_residual(x)
        return self.relu(hidden_)


class TemporalConvolution(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        padding=4,
    ):
        super(TemporalConvolution, self).__init__()

        self.padding = padding
        self.t_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(self.padding, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        init_weights(self.t_conv, bs=1)
        init_weights(self.bn, bs=1)

    def forward(self, x):
        x = self.bn(self.t_conv(x))
        return x


class StGcnBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        stride=1,
        residual=True,
        temporal_kernel_size=9,
        temporal_padding=-1,
        GraphConv=GraphConvolution,
        TempConv=TemporalConvolution,
    ):
        super(StGcnBlock, self).__init__()
        equal_padding = int((temporal_kernel_size - 1) / 2)
        if temporal_padding < 0:
            temporal_padding = equal_padding
            self.residual_shrink = None
        else:
            assert temporal_padding <= equal_padding
            self.residual_shrink = equal_padding - temporal_padding
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = TempConv(
            out_channels,
            out_channels,
            stride=stride,
            kernel_size=temporal_kernel_size,
            padding=temporal_padding,
        )
        self.relu = nn.ReLU()
        if not residual:
            self.residual = zero
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = unity
        else:
            self.residual = TempConv(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            )

    def forward(self, x):
        z = self.tcn(self.gcn(x))

        if self.residual_shrink:
            # Centered residuals:
            # If temporal zero-padding is removed, the feature-map shrinks at every temporal conv
            # The residual should shrink correspondingly, i.e. (kernel_size - 1) / 2) on each side
            r = self.residual(x[:, :, self.residual_shrink : -self.residual_shrink])
        else:
            r = self.residual(x)

        return self.relu(z + r)


class CoTemporalConvolution(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        extra_delay: int = None,
        bn_momentum=0.1,
        padding=0,
    ):
        super(CoTemporalConvolution, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.t_conv = ConvCo2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(self.padding, 0),
            stride=(stride, 1),
        )
        self.bn = torch.nn.BatchNorm1d(out_channels, momentum=bn_momentum)
        if extra_delay:
            self.extra_delay = Delay(extra_delay)
        init_weights(self.t_conv, bs=1)
        init_weights(self.bn, bs=1)

    def forward(self, x):
        x = self.t_conv(x)
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


class CoStGcnBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        stride=1,
        residual=True,
        bn_momentum=0.1,
        t_padding=0,
        GraphConv=GraphConvolution,
        CoTempConv=CoTemporalConvolution,
    ):
        super(CoStGcnBlock, self).__init__()
        self.stride = stride
        self.gcn = unsqueezed(GraphConv(in_channels, out_channels, A, bn_momentum))
        self.tcn = CoTempConv(
            out_channels, out_channels, stride=stride, padding=t_padding
        )
        self.relu = torch.nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (self.stride == 1):
            self.residual = Delay(self.tcn.kernel_size // 2, temporal_fill="zeros")
        else:
            self.residual = CoTempConv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=self.stride,
                extra_delay=(self.tcn.kernel_size // 2) // self.stride,
            )

    def forward(self, x):
        z = self.tcn(self.gcn(x))
        r = self.residual(x)
        if type(z) is TensorPlaceholder:
            return TensorPlaceholder(z.shape)
        return self.relu(z + r)

    def forward_clip(self, x):
        NM, C, T, V = x.shape
        res = []
        for t in range(T):
            z = self.tcn(self.gcn(x[:, :, t]))
            r = self.residual(x[:, :, t])
            if type(z) is not TensorPlaceholder:
                res.append(self.relu(z + r))
        return torch.stack(res, dim=2)

    @property
    def delay(self):
        return self.tcn.delay
