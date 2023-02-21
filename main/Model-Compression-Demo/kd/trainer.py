# @Time : 2020-04-19 13:14 
# @Author : Ben 
# @Version：V 0.1
# @File : trainer.py
# @desc :训练老师跟学生网络

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
from nets import TeacherNet, StudentNet
from torch import nn
from torch.functional import F


class Trainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # 启用GPU
        self.teacher_net = TeacherNet().to(self.device) # 导入到GPU
        self.student_net = StudentNet().to(self.device) # 导入到GPU
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ])
        self.train_data = DataLoader(datasets.MNIST("../datasets/", train=True, transform=self.trans, download=False),
                                     batch_size=1000, shuffle=True, drop_last=True)
        self.test_data = DataLoader(datasets.MNIST("../datasets/", False, self.trans, download=False), batch_size=10000,
                                    shuffle=True)
        self.CrossEntropyLoss = nn.CrossEntropyLoss() # 损失函数-交叉熵
        self.KLDivLoss = nn.KLDivLoss() # 损失函数-相对熵
        self.teacher_optimizer = torch.optim.Adam(self.teacher_net.parameters())
        self.student_optimizer = torch.optim.Adam(self.student_net.parameters())

    def train_teacher(self):
        self.teacher_net.train()
        # 设置模型处于train阶段
        # 因为网络含有Dropout层，该层在train和eval两个阶段表现不同，所以需要明确区分
        for epoch in range(1, 3):
            total = 0
            for i, (data, label) in enumerate(self.train_data):
                data, label = data.to(self.device), label.to(self.device)
                output = self.teacher_net(data) # 根据输入data计算得到output
                loss = self.CrossEntropyLoss(output, label) # 根据output计算交叉熵loss
                self.teacher_optimizer.zero_grad() # 将模型的参数梯度初始化为0
                loss.backward() # 反向传播计算梯度
                self.teacher_optimizer.step()  # 更新所有参数   

                total += len(data)
                progress = math.ceil(i / len(self.train_data) * 50)
                print("\rTrain teacher_net epoch %d: %d/%d, [%-51s] %d%%" %
                      (epoch, total, len(self.train_data.dataset),
                       '-' * progress + '>', progress * 2), end='')
            # torch.save(self.teacher_net.state_dict(), "models/teacher_net.pth")
            self.evaluate(self.teacher_net)

    def train_student(self):
        self.student_net.train()
        for epoch in range(1, 3):
            total = 0
            for i, (data, label) in enumerate(self.train_data):
                data, label = data.to(self.device), label.to(self.device)
                teacher_output = self.teacher_net(data) # 计算出一个输出
                student_output = self.student_net(data) # 计算出一个输出
                teacher_output = teacher_output.detach() #？
                loss = self.distillation(student_output, label, teacher_output, temp=5.0, alpha=0.7)
                # 这是总的loss，Total loss = distillation loss × β + student loss × α
                # loss = self.KLDivLoss(student_output, teacher_output)
                # loss = self.CrossEntropyLoss(output, label)

                # 学生模型根据所得Loss训练
                self.student_optimizer.zero_grad()
                loss.backward()
                self.student_optimizer.step()

                total += len(data)
                progress = math.ceil(i / len(self.train_data) * 50)
                print("\rTrain student_net epoch %d: %d/%d, [%-51s] %d%%" %
                      (epoch, total, len(self.train_data.dataset),
                       '-' * progress + '>', progress * 2), end='')
            # torch.save(self.student_net.state_dict(), "models/student_net.pth")
            self.evaluate(self.student_net)

    def evaluate(self, net):
        self.teacher_net.eval()
        # 设置模型处于eval阶段
        for data, label in self.test_data:
            data, label = data.to(self.device), label.to(self.device)
            output = net(data)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            acc = pred.eq(label.view_as(pred)).sum().item() / self.test_data.batch_size
            print(f"\nevaluate acc:{acc * 100:2f}%")

    def distillation(self, y, labels, teacher_scores, temp, alpha):
        distllation_loss = self.KLDivLoss(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1))
        student_loss = F.cross_entropy(y, labels)
        return  distllation_loss * (temp * temp * 2.0 * alpha) + student_loss * (1. - alpha)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train_teacher()
    trainer.train_student()
