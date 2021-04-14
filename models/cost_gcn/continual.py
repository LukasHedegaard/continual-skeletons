import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from enum import Enum
from torch.nn import Module
from functools import wraps
from typing import Callable
from torch.nn.modules.conv import (
    _ConvNd,
    _reverse_repeat_tuple,
    _size_1_t,
    _single,
)
from torch.nn.modules.pooling import _AvgPoolNd, AdaptiveAvgPool1d

from ride.utils.logging import getLogger


class FillMode(Enum):
    REPLICATE = "replicate"
    ZEROS = "zeros"


State = Tuple[Tensor, int]

logger = getLogger(__name__, log_once=True)


def unsqueezed(instance: Module, dim: int = 2):
    def decorator(func: Callable[[Tensor], Tensor]):
        @wraps(func)
        def call(x: Tensor) -> Tensor:
            x = x.unsqueeze(dim)
            x = func(x)
            x = x.squeeze(dim)
            return x

        return call

    instance.forward_regular = instance.forward
    instance.forward = decorator(instance.forward)

    return instance


class ConvCo1d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: FillMode = "zeros",
        temporal_fill: FillMode = "replicate",
    ):
        r"""Applies a continual 1D convolution over an input signal composed of several input
        planes.

        Assuming an input of shape `(B, C, T)`, it computes the convolution over one temporal instant `t` at a time
        where `t` ∈ `range(T)`, and keeps an internal state. Two forward modes are supported here.

        `forward`   takes an input of shape `(B, C)`, and computes a single-frame output (B, C') based on its internal state.
                    On the first execution, the state is initialised with either ``'zeros'`` (corresponding to a zero padding of kernel_size[0]-1)
                    or with a `'replicate'`` of the first frame depending on the choice of `temporal_fill`.
                    `forward` also supports a functional-style exercution, by passing a `prev_state` explicitely as parameters, and by
                    optionally returning the updated `next_state` via the `return_next_state` parameter.
                    NB: The output when recurrently applying forward will be delayed by the `kernel_size[0] - 1`.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. NB: stride > 1 over the first channel is not supported. Default: 1
            padding (int or tuple, optional): Zero-padding added to all three sides of the input. NB: padding over the first channel is not supported. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. NB: dilation > 1 over the first channel is not supported. Default: 1
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
            temporal_fill (string, optional): ``'zeros'`` or ``'replicate'`` (= "boring video"). `temporal_fill` determines how state is initialised and which padding is applied during `forward_regular` along the temporal dimension. Default: ``'replicate'``

        Attributes:
            weight (Tensor): the learnable weights of the module of shape
                            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                            The values of these weights are sampled from
                            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
            bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                            then the values of these weights are
                            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
            state (List[Tensor]):  a running buffer of partial computations from previous frames which are used for
                            the calculation of subsequent outputs.

        """
        kernel_size = _single(kernel_size)

        padding = _single(padding)
        if padding[0] != 0:
            logger.debug(
                "Padding along the temporal dimension only affects the computation in `forward_regular`. In `forward` it is omitted."
            )

        stride = _single(stride)
        if stride[0] > 1:
            logger.warn(
                f"Temporal stride of {stride[0]} will result in repeated outputs every {stride[0]-1} / {stride[0]} steps"
            )

        dilation = _single(dilation)
        assert dilation[0] == 1, "Temporal dilation > 1 is not supported currently."

        super(ConvCo1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed=False,
            output_padding=_single(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
            (self.kernel_size[0] - 1,), 2
        )

        assert temporal_fill in {"zeros", "replicate"}
        self.make_padding = {"zeros": torch.zeros_like, "replicate": torch.clone}[
            temporal_fill
        ]
        # init_state is called in `_forward`

    def init_state(
        self,
        first_output: Tensor,
    ) -> State:
        padding = self.make_padding(first_output)
        state_buffer = padding.repeat(self.kernel_size[0] - 1, 1, 1, 1)
        state_index = 0
        stride_index = 0
        if not hasattr(self, "state_buffer"):
            self.register_buffer("state_buffer", state_buffer, persistent=False)
        return state_buffer, state_index, stride_index

    def clean_state(self):
        self.state_buffer = None
        self.state_index = None
        self.stride_index = None

    def get_state(self):
        if (
            hasattr(self, "state_buffer")
            and self.state_buffer is not None
            and hasattr(self, "state_index")
            and self.state_index is not None
            and hasattr(self, "stride_index")
            and self.stride_index is not None
        ):
            return (self.state_buffer, self.state_index, self.stride_index)
        else:
            return None

    @staticmethod
    def from_regular(
        module: torch.nn.Conv3d, temporal_fill: FillMode = "replicate"
    ) -> "ConvCo1d":
        # padding = (0, *module.padding[1:])
        dilation = (1, *module.dilation[1:])
        if dilation != module.dilation:
            logger.warning(
                f"Using dilation = {dilation} for ConvCo1d (converted from {module.dilation})"
            )

        rmodule = ConvCo1d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            temporal_fill=temporal_fill,
        )
        with torch.no_grad():
            rmodule.weight.copy_(module.weight)
            if module.bias is not None:
                rmodule.bias.copy_(module.bias)
        return rmodule

    def forward(self, input: Tensor, update_state=True) -> Tensor:
        output, (
            new_buffer,
            new_state_index,
            new_stride_index,
        ) = self._forward(input, self.get_state())
        if update_state:
            self.state_buffer = new_buffer
            self.state_index = new_state_index
            self.stride_index = new_stride_index
        return output

    def _forward(self, input: Tensor, prev_state: State) -> Tuple[Tensor, State]:
        assert (
            len(input.shape) == 2
        ), "Only a single time instance should be passed at a time."

        # B, C -> B, C, 1
        x = input.unsqueeze(2)

        x = F.conv1d(
            input=x,
            weight=self.weight,
            bias=None,
            stride=(1,),
            padding=(self.kernel_size[0] - 1,),
            dilation=self.dilation,
            groups=self.groups,
        )

        x_out, x_rest = x[:, :, 0], x[:, :, 1:]

        # Prepare previous state
        buffer, index, stride_index = prev_state or self.init_state(x_rest)

        tot = len(buffer)
        if stride_index == 0:
            x_out = x_out.clone()
            for i in range(tot):
                x_out += buffer[(i + index) % tot, :, :, tot - i - 1]

            if self.bias is not None:
                x_out += self.bias[None, :]
        else:
            x_out = None

        # Update next state
        next_buffer = buffer.clone() if self.training else buffer.detach()
        next_buffer[index] = x_rest
        next_index = (index + 1) % tot
        next_stride_index = (stride_index + 1) % self.stride[0]

        return x_out, (next_buffer, next_index, next_stride_index)

    def forward_regular(self, input: Tensor):
        assert (
            len(input.shape) == 3
        ), "A tensor of size B,C,T should be passed as input."
        T = input.shape[2]
        self.clean_state()

        pad_start = [self.make_padding(input[:, :, 0]) for _ in range(self.padding[0])]
        inputs = [input[:, :, t] for t in range(T)]
        pad_end = [self.make_padding(input[:, :, -1]) for _ in range(self.padding[0])]

        # Recurrently pass through, updating state
        outs = []
        for t, i in enumerate([*pad_start, *inputs]):
            o, (self.state_buffer, self.state_index, self.stride_index) = self._forward(
                i, self.get_state()
            )
            if self.kernel_size[0] - 1 <= t and o:
                outs.append(o)

        # Don't save state for the end-padding
        tmp_buffer, tmp_index, tmp_stride_index = self.get_state()
        for t, i in enumerate(pad_end):
            o, (tmp_buffer, tmp_index, tmp_stride_index) = self._forward(
                i, (tmp_buffer, tmp_index, tmp_stride_index)
            )
            if o:
                outs.append(o)

        if len(outs) > 0:
            outs = torch.stack(outs, dim=2)
        else:
            outs = torch.tensor([])
        return outs

    @property
    def delay(self):
        return self.kernel_size[0] - 1


# Make sure the flops are counted in `ptflops`
try:
    from ptflops import flops_counter as fc

    fc.MODULES_MAPPING[ConvCo1d] = fc.conv_flops_counter_hook
except Exception as e:
    logger.warn(f"Failed to add flops_counter_hook for ConvCo3d: {e}")


class AvgPoolCo1d(_AvgPoolNd):
    """
    Continual Average Pool in 1D

    Pooling results are stored between `forward` exercutions and used to pool subsequent
    inputs along the temporal dimension with a spacified `window_size`.
    Example: For `window_size = 3`, the two previous results are stored and used for pooling.
    `temporal_fill` determines whether to initialize the state with a ``'replicate'`` of the
    output of the first execution or with with ``'zeros'``.
    """

    def __init__(
        self,
        window_size: int,
        temporal_fill: FillMode = "replicate",
        temporal_dilation: int = 1,
        *args,
        **kwargs,
    ):
        assert window_size > 0
        assert temporal_fill in {"zeros", "replicate"}
        self.window_size = window_size
        self.temporal_dilation = temporal_dilation
        self.make_padding = {"zeros": torch.zeros_like, "replicate": torch.clone}[
            temporal_fill
        ]
        super(AvgPoolCo1d, self).__init__(*args, **kwargs)

        self.temporal_pool = AdaptiveAvgPool1d(1)

        if self.temporal_dilation > 1:
            self.frame_index_selection = torch.tensor(
                range(0, self.window_size, self.temporal_dilation)
            )

        # state is initialised in self.forward

    def init_state(
        self,
        first_output: Tensor,
    ) -> State:
        padding = self.make_padding(first_output)
        state_buffer = torch.stack([padding for _ in range(self.window_size)], dim=0)
        state_index = 0
        if not hasattr(self, "state_buffer"):
            self.register_buffer("state_buffer", state_buffer, persistent=False)
        return state_buffer, state_index

    def clean_state(self):
        self.state_buffer = None
        self.state_index = None

    def get_state(self):
        if (
            hasattr(self, "state_buffer")
            and self.state_buffer is not None
            and hasattr(self, "state_index")
            and self.state_buffer is not None
        ):
            return (self.state_buffer, self.state_index)
        else:
            return None

    def forward(self, input: Tensor) -> Tensor:
        output, (self.state_buffer, self.state_index) = self._forward(
            input, self.get_state()
        )
        return output

    def _forward(
        self,
        input: Tensor,
        prev_state: State,
    ) -> Tuple[Tensor, State]:
        assert len(input.shape) == 2, "Only a single time-instance should be passed."

        if prev_state is None:
            buffer, index = self.init_state(input)
        else:
            buffer, index = prev_state

        buffer[index] = input

        if self.temporal_dilation == 1:
            frame_selection = buffer
        else:
            frame_selection = buffer.index_select(
                dim=0, index=self.frame_index_selection
            )

        _, B, C = frame_selection.shape
        pooled_window = self.temporal_pool(
            frame_selection.permute(1, 2, 0)  # T, B, C -> B, C, T
        ).reshape(B, C)
        new_index = (index + 1) % self.window_size
        new_buffer = buffer.clone() if self.training else buffer.detach()

        return pooled_window, (new_buffer, new_index)

    def forward_regular(self, input: Tensor):
        """If input.shape[2] == self.window_size, a global pooling along temporal dimension is performed
        Otherwise, the pooling is performed per frame
        """
        assert (
            len(input.shape) == 3
        ), "A tensor of size B,C,T should be passed as input."

        outs = []
        for t in range(input.shape[2]):
            o = self.forward(input[:, :, t])
            if self.window_size - 1 <= t:
                outs.append(o)

        if len(outs) == 0:
            return torch.tensor([])

        if input.shape[2] == self.window_size:
            # In order to be compatible with downstream forward_regular, select only last frame
            # This corrsponds to the regular global pool
            return outs[-1].unsqueeze(2)

        else:
            return torch.stack(outs, dim=2)
