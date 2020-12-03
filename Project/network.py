# coding:utf-8

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class Lin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(Lin, self).__init__()
        image_size = 28
        self.input_size = image_size * image_size
        self.output_size = 10
        self.batch_size = 64

        self.W = nn.Parameter(torch.zeros(size=(self.input_size, self.output_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.b = nn.Parameter(torch.zeros(size=(self.output_size, 1)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, x):
        data = x.reshape(self.batch_size, -1)  # 64*784
        hidden_output = F.linear(data, self.W.t(), self.b.reshape(self.output_size))

        output = F.log_softmax(hidden_output, dim=1)  # 64*10
        return output


class Full(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(Full, self).__init__()
        self.input_size = 28 * 28
        self.hidden_size = 300
        self.output_size = 10
        self.batch_size = 64

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        data = x.reshape(self.batch_size, -1)
        hidden_output = torch.tanh(self.fc1(data))
        # hidden_output = F.relu(self.fc1(data))
        output = self.fc2(hidden_output)
        output = F.log_softmax(output, dim=1)  # 64*10
        return output


class Conv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(Conv, self).__init__()
        self.num_classes = 10
        self.layer1 = nn.Sequential(  
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(16),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2, stride=2))  

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),  # 14+2*2-5+1=14, output_size = 14*14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 14/2=7

        self.fc = nn.Linear(7*7*32, self.num_classes)

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        output = F.log_softmax(output, dim=1)
        return output
