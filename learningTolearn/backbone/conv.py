# -*- coding: utf-8 -*-
"""
   Description :   Conv model
   Author :        xxm
"""
import torch.nn as nn

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)
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
        ('conv', MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1)),
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
        Number of classes (output of the model).

    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.

    feature_size : int (default: 64)
        Number of features returned by the convolutional head.

    flatten: bool (default: True)
        Flatten feature map under episodic training.
        if False: input will accept meta-task. [batch, task, channel, width, height]

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """

    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64, flatten=True):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.flatten = flatten

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size)),
            ('layer2', conv_block(hidden_size, hidden_size)),
            ('layer3', conv_block(hidden_size, hidden_size)),
            ('layer4', conv_block(hidden_size, hidden_size))
        ]))

        self.classifier = MetaLinear(feature_size, out_features)

    def forward(self, inputs, params=None):
        if self.flatten:
            # [batch_size, 3, 84, 84]
            features = self.features(inputs, params=get_subdict(params, 'features'))
            # [batch_size, hidden_size, 5, 5]
            features = features.view(features.size(0), -1)
            logits = self.classifier(features, params=get_subdict(params, 'classifier'))
            return logits
        else:
            # [batch, task, channel, width, height]
            # inputs: [16, 25, 3, 84, 84]

            # Train before embedding [400, 3, 84, 84]
            embeddings = self.features(inputs.view(-1, *inputs.shape[2:]))
            # Train after embedding: [400, 64, 5, 5]

            return embeddings.view(*inputs.shape[:2], -1)  # [16, 25, 64x5x5]


def ModelConvOmniglot(out_features, hidden_size=64, flatten=True):
    return MetaConvModel(1, out_features,
                         hidden_size=hidden_size, feature_size=hidden_size,
                         flatten=flatten)


def ModelConv(out_features, hidden_size=64, flatten=True):
    return MetaConvModel(3, out_features,
                         hidden_size=hidden_size, feature_size=5 * 5 * hidden_size,
                         flatten=flatten)


def _meta_model_without_flatten_test():
    """
    1. input: episodic training
    2. feature: get meta-task feature_map without flatten
    """
    import torch

    input = torch.rand(16, 25, 3, 84, 84)
    model = MetaConvModel(3, 5, hidden_size=64, feature_size=5 * 5 * 64, flatten=False)
    out = model(input)
    print(out.shape)


def _model_flatten_test():
    """
    1. input: [b, c, h, w]
    2. get weight and bias like `maml`
    3. return : [batch_size, num_classes]
    """
    import torch

    input = torch.rand(32, 3, 84, 84)
    model = MetaConvModel(3, 5, hidden_size=64, feature_size=5 * 5 * 64, flatten=True)
    out = model(input)
    print(out.shape)


if __name__ == '__main__':
    _model_flatten_test()
    # _meta_model_without_flatten_test()
