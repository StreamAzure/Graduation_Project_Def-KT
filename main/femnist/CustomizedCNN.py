import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CustomizedCNN(nn.Module):
    def __init__(self):
        super(CustomizedCNN, self).__init__()
        #输入为MNIST，张量(1,28,28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        #经过卷积层后，张量(32,26,26)
        #再经过2*2的最大池化层，张量(32,13,13)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        #经过卷积层后，张量(64,11,11)
        #再经过2*2的最大池化层，张量(64,5,5)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        #经过卷积层后，张量(64,3,3)
        #64*3*3=576
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.conv3(x)
        # 展平张量
        x = x.view(-1, 576) # 576 = 64 * 3 * 3
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print("forward finished")
        return x
        