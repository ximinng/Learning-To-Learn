# -*- coding: utf-8 -*-
"""
   Description :   train model
   Author :        xxm
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import json

import torchvision
import torchvision.transforms as transforms

# load params
with open("./params.json", 'r') as f:
    params = json.load(f)

print(params['epochs'])

# Composes several transforms together
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # Normalize a tensor image with mean and standard deviation
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

train_set = torchvision.datasets.CIFAR10(root='../data', train=True,
                                         download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                          shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Create Model and DataParallel

from model import Net

net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print(''.join('%d GPUs is available!' % torch.cuda.device_count()))

    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
else:
    print("no GPUs found,use CPU instead!")

net.to(device)

# Define a Loss function and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                      lr=params['learning_rate'],
                      momentum=params['momentum'])

# Train the network on the train data

for epoch in range(params['epochs']):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, start=0):
        # get the inputs; data is a list of [inputs, labels]
        # if torch.cuda.device_count() > 1:
        #     inputs, labels = data
        #     inputs = inputs.to(device)
        #     labels = labels.to(device)
        # else:
        #     inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs).to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Test the network on the test data

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        # if torch.cuda.device_count() > 1:
        #     images, labels = data
        #     images = images.to(device)
        #     labels = labels.to(device)
        # else:
        #     images, labels = data

        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
