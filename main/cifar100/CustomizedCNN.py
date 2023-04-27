import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CustomizedCNN(nn.Module):
    def __init__(self, channels=32):
        super(CustomizedCNN, self).__init__()
        self.num_channels = channels
        #输入为CIFAR100，张量(3,32,32)
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1)
        #经过卷积层后，张量(32,30,30)
        #再经过2*2的最大池化层，张量(32,15,15)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1)
        #经过卷积层后，张量(64,13,13)
        #再经过2*2的最大池化层，张量(64,6,6)
        self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 2, 3, stride=1)
        #经过卷积层后，张量(64,4,4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(4 * 4 * self.num_channels * 2, self.num_channels * 4)
        self.fc2 = nn.Linear(self.num_channels * 4, 100)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.conv3(x)
        # 展平张量
        x = x.view(-1, 4 * 4 * self.num_channels * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        