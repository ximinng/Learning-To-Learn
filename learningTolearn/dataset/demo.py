# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
from torchmeta.datasets import MiniImagenet
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader

dataset = MiniImagenet("/few-shot-datasets",
                       # Number of ways
                       num_classes_per_task=5,
                       # Resize the images to 28x28 and converts them to PyTorch tensors (from Torchvision)
                       transform=Compose([Resize(84), ToTensor()]),
                       # Transform the labels to integers (e.g. ("Glagolitic/character01", "Sanskrit/character14", ...) to (0, 1, ...))
                       target_transform=Categorical(num_classes=5),
                       # Creates new virtual classes with rotated versions of the images (from Santoro et al., 2016)
                       class_augmentations=[Rotation([90, 180, 270])],
                       meta_train=True,
                       download=True)
dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=5, num_test_per_class=15)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

for batch in dataloader:
    train_inputs, train_targets = batch["train"]
    print('Train inputs shape: {0}'.format(train_inputs.shape))  # torch.Size([16, 25, 3, 84, 84])
    print('Train targets shape: {0}'.format(train_targets.shape))  # torch.Size([16, 25])

    test_inputs, test_targets = batch["test"]
    print('Test inputs shape: {0}'.format(test_inputs.shape))  # torch.Size([16, 75, 3, 84, 84])
    print('Test targets shape: {0}'.format(test_targets.shape))  # torch.Size([16, 75])

