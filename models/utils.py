import math

import torch.nn as nn
from numpy import prod
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd


def init_weights(module_, bs=1):
    if isinstance(module_, _ConvNd):
        nn.init.constant_(module_.bias, 0)
        if bs == 1:
            nn.init.kaiming_normal_(module_.weight, mode="fan_out")
        else:
            nn.init.normal_(
                module_.weight,
                0,
                math.sqrt(2.0 / (prod(module_.weight.size()) * bs)),
            )
    elif isinstance(module_, _BatchNorm):
        nn.init.constant_(module_.weight, bs)
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Linear):
        nn.init.normal_(module_.weight, 0, math.sqrt(2.0 / bs))


def zero(x):
    return 0


def unity(x):
    return x
