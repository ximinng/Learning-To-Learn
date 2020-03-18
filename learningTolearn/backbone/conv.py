# -*- coding: utf-8 -*-
"""
   Description :   Conv and MLP model
   Author :        xxm
"""
import torch.nn as nn

from collections import OrderedDict
from torchmeta.modules import MetaModule, MetaConv2d, MetaBatchNorm2d, MetaSequential, MetaLinear
from torchmeta.modules.utils import get_subdict

"""
MetaConvModel(
  (features): MetaSequential(
    (layer1): MetaSequential(
      (conv): MetaConv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm): MetaBatchNorm2d(64, eps=1e-05, momentum=1.0, affine=True, track_running_stats=False)
      (relu): ReLU()
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (layer2): MetaSequential(
      (conv): MetaConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm): MetaBatchNorm2d(64, eps=1e-05, momentum=1.0, affine=True, track_running_stats=False)
      (relu): ReLU()
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (layer3): MetaSequential(
      (conv): MetaConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm): MetaBatchNorm2d(64, eps=1e-05, momentum=1.0, affine=True, track_running_stats=False)
      (relu): ReLU()
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (layer4): MetaSequential(
      (conv): MetaConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm): MetaBatchNorm2d(64, eps=1e-05, momentum=1.0, affine=True, track_running_stats=False)
      (relu): ReLU()
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
  )
  (classifier): MetaLinear(in_features=1600, out_features=5, bias=True)
)
"""


def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))


class MetaConvModel(MetaModule):
    """4-layer Convolutional Neural Network architecture from [1].

    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.

    out_features : int
        Number of classes (num of ways in few-shot set).

    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.

    feature_size : int (default: 64)
        Number of features returned by the convolutional head.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """

    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True))
        ]))
        self.classifier = MetaLinear(feature_size, out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        return logits


def ModelConvOmniglot(out_features, hidden_size=64):
    return MetaConvModel(1, out_features, hidden_size=hidden_size, feature_size=hidden_size)


def ModelConvMiniImagenet(out_features, hidden_size=64):
    return MetaConvModel(3, out_features, hidden_size=hidden_size, feature_size=5 * 5 * hidden_size)
