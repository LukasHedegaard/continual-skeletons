import torch
from torch import Tensor
from typing import Tuple, Union
from torch.nn.modules.pooling import (
    _AvgPoolNd,
    AdaptiveAvgPool1d,
    AdaptiveMaxPool1d,
    AvgPool1d,
    MaxPool1d,
)

from ride.utils.logging import getLogger
from .utils import FillMode


State = Tuple[Tensor, int]
Pool1D = Union[AvgPool1d, MaxPool1d, AdaptiveAvgPool1d, AdaptiveMaxPool1d]


logger = getLogger(__name__)

__all__ = [
    "AvgPoolCo1d",
    "AvgPoolCo2d",
    "MaxPoolCo2d",
    "AdaptiveAvgPoolCo2d",
    "AdaptiveMaxPoolCo2d",
]

State = Tuple[Tensor, int]


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


def RecursivelyWindowPooled(cls: Pool1D) -> torch.nn.Module:  # noqa: C901
    """Wraps a pooling module to create a recursive version which pools across execusions

    Args:
        cls (Pool1D): A 1D pooling Module
    """
    assert cls in {AdaptiveAvgPool1d, MaxPool1d, AvgPool1d, AdaptiveMaxPool1d}

    class RePooled(cls):
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
            super(RePooled, self).__init__(*args, **kwargs)

            self.temporal_pool = (
                AdaptiveAvgPool1d
                if "avg" in str(cls.__name__).lower()
                else AdaptiveMaxPool1d
            )(1)

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
            state_buffer = torch.stack(
                [padding for _ in range(self.window_size)], dim=0
            )
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
            assert (
                len(input.shape) == 3
            ), "Only a single frame should be passed at a time."

            pooled_frame = super(RePooled, self).forward(input)

            if prev_state is None:
                buffer, index = self.init_state(pooled_frame)
            else:
                buffer, index = prev_state

            buffer[index] = pooled_frame

            if self.temporal_dilation == 1:
                frame_selection = buffer
            else:
                frame_selection = buffer.index_select(
                    dim=0, index=self.frame_index_selection
                )

            # Pool along temporal dimension
            T, B, C, S = frame_selection.shape
            x = frame_selection.permute(1, 3, 2, 0)  # B, S, C, T
            x = x.reshape(B * S, C, T)
            x = self.temporal_pool(x)
            x = x.reshape(B, S, C)
            x = x.permute(0, 2, 1)  # B, C, S
            pooled_window = x

            new_index = (index + 1) % self.window_size
            new_buffer = buffer.clone() if self.training else buffer.detach()

            return pooled_window, (new_buffer, new_index)

        def forward_regular(self, input: Tensor):
            """If input.shape[2] == self.window_size, a global pooling along temporal dimension is performed
            Otherwise, the pooling is performed per frame
            """
            assert (
                len(input.shape) == 4
            ), "A tensor of size B,C,T,S should be passed as input."

            outs = []
            for t in range(input.shape[2]):
                o = self.forward(input[:, :, t])
                if self.window_size - 1 <= t:
                    outs.append(o)

            if len(outs) == 0:
                return torch.tensor([])

            if input.shape[2] == self.window_size:
                # In order to be compatible with downstream forward3d, select only last frame
                # This corrsponds to the regular global pool
                return outs[-1].unsqueeze(2)

            else:
                return torch.stack(outs, dim=2)

    RePooled.__doc__ = f"""
    Recursive {cls.__name__}

    Pooling results are stored between `forward` exercutions and used to pool subsequent
    inputs along the temporal dimension with a spacified `window_size`.
    Example: For `window_size = 3`, the two previous results are stored and used for pooling.
    `temporal_fill` determines whether to initialize the state with a ``'replicate'`` of the
    output of the first execution or with with ``'zeros'``.

    Parent doc:
    {cls.__doc__}
    """

    return RePooled


AvgPoolCo2d = RecursivelyWindowPooled(AvgPool1d)
MaxPoolCo2d = RecursivelyWindowPooled(MaxPool1d)
AdaptiveAvgPoolCo2d = RecursivelyWindowPooled(AdaptiveAvgPool1d)
AdaptiveMaxPoolCo2d = RecursivelyWindowPooled(AdaptiveMaxPool1d)
