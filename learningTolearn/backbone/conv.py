# -*- coding: utf-8 -*-
"""
   Description :   Conv model
   Author :        xxm
"""
import torch.nn as nn

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)


def conv_block(in_channels, out_channels, bias=True,
               activation=nn.ReLU(inplace=True),
               use_dropout=False, p=0.1):
    res = MetaSequential(OrderedDict([
        ('conv', MetaConv2d(int(in_channels), int(out_channels), kernel_size=3, padding=1, bias=bias)),
        ('norm', MetaBatchNorm2d(int(out_channels), momentum=1., track_running_stats=False)),
        ('relu', activation),
        ('pool', nn.MaxPool2d(2)),
    ]))

    if use_dropout:
        res.add_module('dropout', nn.Dropout2d(p))

    return res


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

    embedding: bool (default: True)
        Flatten feature map under episodic training.
        if False: input will accept meta-task. [batch/task, num of pic, channel, width, height]
        for prototype network.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """

    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64, embedding=True):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.embedding = embedding

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size)),
            ('layer2', conv_block(hidden_size, hidden_size)),
            ('layer3', conv_block(hidden_size, hidden_size)),
            ('layer4', conv_block(hidden_size, hidden_size))
        ]))

        self.classifier = MetaLinear(feature_size, out_features)

    def forward(self, inputs, params=None):
        if self.embedding:
            # inputs shape: [batch, task, channel, width, height]
            # like: [16, 25, 3, 84, 84]

            # Train before embedding [400, 3, 84, 84]
            embeddings = self.features(inputs.view(-1, *inputs.shape[2:]))
            # Train after embedding: [400, 64, 5, 5]

            return embeddings.view(*inputs.shape[:2], -1)  # [16, 25, 64x5x5]
        else:  # MAML
            # inputs shape: [batch_size, channel, width, height]
            features = self.features(inputs, params=self.get_subdict(params, 'features'))
            features = features.view(features.size(0), -1)
            logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
            return logits


def ModelConvOmniglot(out_features, hidden_size=64, flatten=True):
    return MetaConvModel(1, out_features,
                         hidden_size=hidden_size, feature_size=hidden_size,
                         embedding=flatten)


def ModelConv(out_features, hidden_size=64, flatten=True):
    return MetaConvModel(3, out_features,
                         hidden_size=hidden_size, feature_size=5 * 5 * hidden_size,
                         embedding=flatten)


class EmbeddingImagenet(nn.Module):
    """4-layer Convolutional Neural Network architecture from [1].

    Parameters
    ----------
    emb_size : int
        embedding space after backbone trained.

    References
    ----------
    .. [1] [Kim et al. 2019] Kim, J.; Kim, T.; Kim, S.; and Yoo, C. D. (2019).
        Edge-labeling graph neural network for few-shot learning. In CVPR.
    """

    def __init__(self, emb_size):
        super(EmbeddingImagenet, self).__init__()
        self.in_channels = 3
        self.hidden = 64
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        self.layers = nn.Sequential(OrderedDict([
            ('layer1', conv_block(self.in_channels, self.hidden, bias=False,
                                  activation=nn.LeakyReLU(0.2))),
            ('layer2', conv_block(self.hidden, self.hidden * 1.5, bias=False,
                                  activation=nn.LeakyReLU(0.2))),
            ('layer3', conv_block(self.hidden * 1.5, self.hidden * 2, bias=False,
                                  activation=nn.LeakyReLU(0.2),
                                  use_dropout=True, p=0.4)),
            ('layer4', conv_block(self.hidden * 2, self.hidden * 4, bias=False,
                                  activation=nn.LeakyReLU(0.2),
                                  use_dropout=True, p=0.5))
        ]))
        self.last_layer = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                                  out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, x):
        features = self.layers(x)
        return self.last_layer(features.view(features.size(0), -1))


def _meta_model_embedding_test():
    """
    1. input: episodic training
    2. feature: get meta-task feature_map without flatten
    """
    import torch

    input = torch.rand(16, 25, 3, 84, 84)
    model = MetaConvModel(3, 5, hidden_size=64, feature_size=5 * 5 * 64, embedding=False)
    out = model(input)
    print(out.shape)


def _meta_model_test():
    """
    1. input: [b, c, h, w]
    2. get weight and bias like `maml`
    3. return : [batch_size, num_classes]
    """
    import torch

    input = torch.rand(32, 3, 84, 84)
    model = MetaConvModel(3, 5, hidden_size=64, feature_size=5 * 5 * 64, embedding=True)
    out = model(input)
    print(out.shape)


def _model_egnn_test():
    import torch

    input = torch.rand(32, 3, 84, 84)
    model = EmbeddingImagenet(555)
    print(model)
    out = model(input)
    print(out.shape)


if __name__ == '__main__':
    _meta_model_test()
    _meta_model_embedding_test()
    # _model_egnn_test()
