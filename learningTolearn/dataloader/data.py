# -*- coding: utf-8 -*-
"""
   Description :   data loader by torchemta
   Author :        xxm
"""
import torch.nn.functional as F
import logging

from collections import namedtuple
from torchmeta.datasets import Omniglot, MiniImagenet, TieredImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose

from learningTolearn.backbone import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid, resnet10, resnet12
from learningTolearn.util import ToTensor1D

Benchmark = namedtuple('Benchmark',
                       'meta_train_dataset meta_val_dataset '
                       'meta_test_dataset model loss_function')


def get_benchmark_by_name(name,
                          folder,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          hidden_size=None):
    """get_benchmark_by_name

      Parameters
      ----------
      name : `str` (like `omniglot`)
          The name of datasets.

      folder : string
          Root directory where the dataset folder `omniglot` exists.

      num_ways : float (default: 0.1)
          Number of classes per task (N in "N-way").

      num_shots : bool (default: False)
          Number of training example per class (k in "k-shot").

      num_shots_test : bool (default: False)
          Number of test example per class.

      hidden_size:
          Number of channels in each convolution layer of the neural network.
      """
    logging.info("datasets: {}".format(name))
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
    if name == 'sinusoid':
        transform = ToTensor1D()

        meta_train_dataset = Sinusoid(num_shots + num_shots_test,
                                      num_tasks=1000000,
                                      transform=transform,
                                      target_transform=transform,
                                      dataset_transform=dataset_transform)
        meta_val_dataset = Sinusoid(num_shots + num_shots_test,
                                    num_tasks=1000000,
                                    transform=transform,
                                    target_transform=transform,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Sinusoid(num_shots + num_shots_test,
                                     num_tasks=1000000,
                                     transform=transform,
                                     target_transform=transform,
                                     dataset_transform=dataset_transform)

        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
        loss_function = F.mse_loss

    elif name == 'omniglot':
        class_augmentations = [Rotation([90, 180, 270])]
        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = Omniglot(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True)
        meta_val_dataset = Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Omniglot(folder,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)

        model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'miniimagenet':
        transform = Compose([Resize(84), ToTensor()])

        meta_train_dataset = MiniImagenet(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_train=True,
                                          dataset_transform=dataset_transform,
                                          download=True)
        meta_val_dataset = MiniImagenet(folder,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_val=True,
                                        dataset_transform=dataset_transform)
        meta_test_dataset = MiniImagenet(folder,
                                         transform=transform,
                                         target_transform=Categorical(num_ways),
                                         num_classes_per_task=num_ways,
                                         meta_test=True,
                                         dataset_transform=dataset_transform)

        # TODO: change backbone here
        model = resnet12(in_channels=3, in_size=(84, 84), num_classes=num_ways)
        # model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
        # logging.info("backbone is: {}".format("resnet 12"))
        loss_function = F.cross_entropy

    elif name == 'tieredimagenet':
        transform = Compose([Resize(84), ToTensor()])

        meta_train_dataset = TieredImagenet(folder,
                                            transform=transform,
                                            target_transform=Categorical(num_ways),
                                            num_classes_per_task=num_ways,
                                            meta_train=True,
                                            dataset_transform=dataset_transform,
                                            download=True)
        meta_val_dataset = TieredImagenet(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_val=True,
                                          dataset_transform=dataset_transform)
        meta_test_dataset = TieredImagenet(folder,
                                           transform=transform,
                                           target_transform=Categorical(num_ways),
                                           num_classes_per_task=num_ways,
                                           meta_test=True,
                                           dataset_transform=dataset_transform)

        # model = resnet12()
        print("backbone is resnet")
        model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)
