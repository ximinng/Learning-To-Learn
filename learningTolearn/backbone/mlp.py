# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
import torch.nn as nn

from collections import OrderedDict
from torchmeta.modules import MetaModule, MetaConv2d, MetaBatchNorm2d, MetaSequential, MetaLinear
from torchmeta.modules.utils import get_subdict


class MetaMLPModel(MetaModule):
    """Multi-layer Perceptron architecture from [1].

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of classes (num of ways in few-shot set).

    hidden_sizes : list of int
        Size of the intermediate representations. The length of this list
        corresponds to the number of hidden layers.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """

    def __init__(self, in_features, out_features, hidden_sizes):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes
        self.features = MetaSequential(OrderedDict([
            ('layer{0}'.format(i + 1), MetaSequential(OrderedDict([
                ('linear', MetaLinear(hidden_size, layer_sizes[i + 1], bias=True)),
                ('relu', nn.ReLU())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])
        ]))
        self.classifier = MetaLinear(hidden_sizes[-1], out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        return logits


def ModelMLPSinusoid(hidden_sizes=[40, 40]):
    return MetaMLPModel(1, 1, hidden_sizes)


if __name__ == '__main__':
    model = ModelMLPSinusoid()
