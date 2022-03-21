import ride  # isort:skip
from collections import OrderedDict

import continual as co
from torch import Tensor, nn

from datasets import datasets
from models.base import CoModelBase, CoSpatioTemporalBlock
from models.s_tr.s_tr import GcnUnitAttention


class CoSTr(
    ride.RideModule,
    ride.TopKAccuracyMetric(1, 3, 5),
    ride.optimizers.SgdOneCycleOptimizer,
    datasets.GraphDatasets,
    CoModelBase,
):
    def __init__(self, hparams):
        # Shapes from Dataset:
        # num_channels, num_frames, num_vertices, num_skeletons
        (C_in, T, V, S) = self.input_shape
        A = self.graph.A

        def CoGcnUnitAttention(in_channels, out_channels, A, bn_momentum=0.1):
            return co.forward_stepping(
                GcnUnitAttention(in_channels, out_channels, A, bn_momentum, num_point=V)
            )

        # fmt: off
        self.layers = co.Sequential(OrderedDict([
            ("layer1", CoSpatioTemporalBlock(C_in, 64, A, padding="equal", window_size=T, residual=False)),
            ("layer2", CoSpatioTemporalBlock(64, 64, A, padding="equal", window_size=T - 1 * 8)),
            ("layer3", CoSpatioTemporalBlock(64, 64, A, padding="equal", window_size=T - 2 * 8)),
            ("layer4", CoSpatioTemporalBlock(64, 64, A, CoGraphConv=CoGcnUnitAttention, padding="equal", window_size=T - 3 * 8)),
            ("layer5", CoSpatioTemporalBlock(64, 128, A, CoGraphConv=CoGcnUnitAttention, padding="equal", window_size=T - 4 * 8, stride=2)),
            ("layer6", CoSpatioTemporalBlock(128, 128, A, CoGraphConv=CoGcnUnitAttention, padding="equal", window_size=(T - 4 * 8) / 2 - 1 * 8)),
            ("layer7", CoSpatioTemporalBlock(128, 128, A, CoGraphConv=CoGcnUnitAttention, padding="equal", window_size=(T - 4 * 8) / 2 - 2 * 8)),
            ("layer8", CoSpatioTemporalBlock(128, 256, A, CoGraphConv=CoGcnUnitAttention, padding="equal", window_size=(T - 4 * 8) / 2 - 3 * 8, stride=2)),
            ("layer9", CoSpatioTemporalBlock(256, 256, A, CoGraphConv=CoGcnUnitAttention, padding="equal", window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 1 * 8)),
            ("layer10", CoSpatioTemporalBlock(256, 256, A, CoGraphConv=CoGcnUnitAttention, padding="equal", window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 2 * 8)),
        ]))
        # fmt: on

        # Other layers defined in CoModelBase.on_init_end

    def map_state_dict(
        self,
        state_dict: "OrderedDict[str, Tensor]",
        strict: bool = True,
    ) -> "OrderedDict[str, Tensor]":
        def map_key(k: str):
            # Handle "layers.layer2.0.1.gcn.g_conv.0.weight" -> "layers.layer2.gcn.g_conv.0.weight"
            k = k.replace("0.1.", "")

            # Handle "layers.layer8.0.0.residual.t_conv.weight" ->layers.layer8.residual.t_conv.weight'
            k = k.replace("0.0.residual", "residual")
            return k

        long_keys = nn.Module.state_dict(self, keep_vars=True).keys()

        if len(long_keys - state_dict.keys()):
            short2long = {map_key(k): k for k in long_keys}
            state_dict = OrderedDict(
                [
                    (short2long[k], v)
                    for k, v in state_dict.items()
                    if strict or k in short2long
                ]
            )
        return state_dict

    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, Tensor]",
        strict: bool = True,
    ):
        state_dict = self.map_state_dict(state_dict, strict)
        return nn.Module.load_state_dict(self, state_dict, strict)

    def map_loaded_weights(self, file, loaded_state_dict):
        return self.map_state_dict(loaded_state_dict)


if __name__ == "__main__":  # pragma: no cover
    ride.Main(CoSTr).argparse()
