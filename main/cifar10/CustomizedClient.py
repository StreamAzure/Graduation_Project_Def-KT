import time
import copy
from torch import nn

from easyfl.client.base import BaseClient
import  torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import accuracy
# from local_cnn import CNNModel
from simple_cnn import Model
from easyfl.tracking import metric
from easyfl.tracking.evaluation import model_size

DML_lr = 0.018 # DML训练的学习率

local_batch_size = 64 # B_1
DML_batch_size = 16 # B_2
# 论文里区分了两个阶段的batch size为B_1和B_2

local_epoch = 5 # M
DML_epoch = 15 # E

alpha = 0.1 # KL散度的权重
beta = 0.9 # 交叉熵的权重

class CustomizedClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(CustomizedClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        # Initialize a classifier for each client.
        # self.local_model= CNNModel()
        self.local_model = Model()
        print("local_batch_size: {}, DML_batch_size: {}, local_epoch: {}, DML_epoch: {}, alpha: {}, beta: {}, DML_lr: {}".format(local_batch_size, DML_batch_size, local_epoch, DML_epoch, alpha, beta, DML_lr))

    def train_local(self, conf, device):
        self.local_model.train()
        self.local_model.to(device)
        local_optimizer = torch.optim.SGD(self.local_model.parameters(),
                                        lr=conf.optimizer.lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay,
                                        nesterov=conf.optimizer.nesterov)
        loss_ce = nn.CrossEntropyLoss()
        train_loader = self.train_data.loader(local_batch_size, self.cid, shuffle=True, seed=conf.seed)
        local_epoch_loss = []
        for epoch in range(local_epoch):
            batch_loss = []
            for data, label in train_loader:
                data, label = data.to(device), label.to(device)
                data, label = Variable(data), Variable(label)
                out = self.local_model(data)
                local_optimizer.zero_grad()
                loss = loss_ce(out, label)

                loss.backward()
                local_optimizer.step()
                batch_loss.append(loss.item())
            local_epoch_loss.append(sum(batch_loss) / len(batch_loss))
        local_loss = sum(local_epoch_loss) / len(local_epoch_loss)
        print("--- local_update_loss : {:.2f}".format(local_loss))
        self.test_local_model(conf, device)

    def train_DML(self, conf, device):
        model_A = copy.deepcopy(self.model)
        model_B = copy.deepcopy(self.local_model)
        model_A.to(device)
        model_B.to(device)
        optimizer_A = torch.optim.SGD(model_A.parameters(),
                                        lr=DML_lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay,
                                        nesterov=conf.optimizer.nesterov)
        optimizer_B = torch.optim.SGD(model_B.parameters(),
                                        lr=DML_lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay,
                                        nesterov=conf.optimizer.nesterov)
        loss_kl = nn.KLDivLoss(reduction='batchmean')
        loss_ce = nn.CrossEntropyLoss()
        train_loader = self.train_data.loader(DML_batch_size, self.cid, shuffle=True, seed=conf.seed)
        
        A_epoch_loss = []
        for epoch in range(DML_epoch):
            A_batch_loss = []
            for data, label in train_loader:
                data, label = data.to(device), label.to(device)
                data, label = Variable(data), Variable(label)

                out_A = model_A(data)
                out_B = model_B(data)

                optimizer_A.zero_grad()
                optimizer_B.zero_grad()

                loss_A = beta * loss_ce(out_A, label) + alpha * loss_kl(F.log_softmax(out_A, dim=1), F.softmax(out_B, dim=1))
                loss_B = beta * loss_ce(out_B, label) + alpha * loss_kl(F.log_softmax(out_B, dim=1), F.softmax(out_A, dim=1))

                loss_A.backward(retain_graph=True)
                loss_B.backward(retain_graph=True)

                optimizer_A.step()
                optimizer_B.step()

                A_batch_loss.append(loss_A.item())
            A_epoch_loss.append(sum(A_batch_loss) / len(A_batch_loss))
        A_loss = sum(A_epoch_loss) / len(A_epoch_loss)
        print("--- DML_update_loss(A model) with Client:{}: {:.2f}".format(self.cid, A_loss))
        # self.local_model = copy.deepcopy(self.model)
        # self.local_model = copy.deepcopy(model_A)
        self.local_model.load_state_dict(model_A.state_dict())
        self.test_local_model(conf, device)
    
    def test_local_model(self, conf, device):
        self.local_model.eval()
        self.local_model = self.local_model.to(device)
        test_loader = self.test_data.loader(100, self.cid, shuffle=False, seed=conf.seed)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                data, label = Variable(data), Variable(label)
                out = self.local_model(data)
                test_loss += F.cross_entropy(out, label, reduction='sum').item()
                pred = out.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)
        # print("--- local_test_loss : {:.2f}".format( test_loss))
        # print("--- local_test_acc : {:.2f}%".format( test_acc))
        return test_loss, test_acc

    def train(self, conf, device, train_local_only):
        """
        每个round中
        1. A组客户端调用一次train，train_local_only为True，单独训练本地模型
        2. B组客户端调用一次train，train_local_only为False，与server分发过来的train过的A组模型进行DML
        """
        if train_local_only:
            self.train_local(conf, device)
        else:
            self.train_DML(conf, device)

    def run_train(self, model, conf, train_local_only):
        self.conf = conf
        if conf.track:
            self._tracker.set_client_context(conf.task_id, conf.round_id, self.cid)
        self._is_train = True
        self.download(model) 
        self.track(metric.TRAIN_DOWNLOAD_SIZE, model_size(model))
        self.decompression()

        self.pre_train()
        self.train(conf, self.device, train_local_only) # 只修改了这里
        self.post_train()

        self.track(metric.TRAIN_ACCURACY, self.train_accuracy)
        self.track(metric.TRAIN_LOSS, self.train_loss)
        self.track(metric.TRAIN_TIME, self.train_time)
        
        if conf.local_test:
            self.test_local()
        self.compression()
        self.track(metric.TRAIN_UPLOAD_SIZE, model_size(self.compressed_model))
        self.encryption()

        return self.upload()
