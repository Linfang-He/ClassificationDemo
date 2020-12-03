# coding:utf-8

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics as metrics
import numpy as np
from torchvision import datasets, transforms
from network import Lin, Full, Conv
from img_transfer import loadImage


def train(data_type, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data, target
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    torch.save(model.state_dict(), data_type+'_params.pkl')


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target  # data: [64, 1, 28, 28]
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # determine index with maximal log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        np.set_printoptions(precision=4, suppress=True)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def demo(data_type, path):
    model = Conv()
    model.load_state_dict(torch.load(data_type+'_conv'+ '_params.pkl'))
    model.eval()
    image_data = loadImage(path)
    image_data = torch.Tensor(image_data)
    image_data = image_data.repeat(64, 1, 1).unsqueeze(1)
    print(image_data.shape)
    output = model(image_data)
    pred = output.argmax(dim=1, keepdim=True)
    return pred[0]


def main():
    net_type = 'full' # 'lin', 'full' or 'conv'
    lr = 0.01  # learning rate
    mom = 0.5  # momentum
    epochs = 10  # number of training epochs

    # define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # fetch and load training data
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, drop_last=True)

    # fetch and load test data
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, drop_last=True)

    # choose network architecture
    if net_type == 'lin':
        net = Lin()
    elif net_type == 'full':
        net = Full()
    else:
        net = Conv()

    if list(net.parameters()):
        # use SGD optimizer
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mom)

        # training and testing loop
        for epoch in range(1, epochs + 1):
            train('digit_' + net_type, net, train_loader, optimizer, epoch)
            test(net, test_loader)
        
if __name__ == '__main__':
    main()
