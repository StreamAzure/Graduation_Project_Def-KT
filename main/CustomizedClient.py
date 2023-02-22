import easyfl
import time
from torch import nn

from easyfl.client.base import BaseClient
import  torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import accuracy
from local_cnn import CNNModel
from easyfl.tracking import metric
from easyfl.tracking.evaluation import model_size

# Inherit BaseClient to implement customized client operations.
class CustomizedClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(CustomizedClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        # Initialize a classifier for each client.
        self.local_model= None
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()
        self.local_model= CNNModel()

    def train_local_only(self, conf, device):
        start_time = time.time()
        loss_fn, optimizers = self.pretrain_setup(conf, device)
        self.train_loss = []
        for i in range(conf.local_epoch): # 注意！！conf.local_epoch按论文说的设为了1
            batch_loss = []
            epoch_acc = []
            for batched_x, batched_y in self.train_loader:
                x, y = batched_x.to(device), batched_y.to(device)
                x, y = Variable(x), Variable(y)
                out = self.local_model(x)
                loss = loss_fn(out, y)
                optimizers[1].zero_grad()
                loss.backward()
                optimizers[1].step()
                batch_loss.append(loss.item())
                epoch_acc.append(accuracy(out, y))
            self.train_loss.append(sum(batch_loss) / len(batch_loss))
            print('Client {} local epoch: {}, loss: {:.4f}'.format(self.cid, i, self.train_loss[-1]))
        # print('Client {} local training time: {:.4f} s'.format(self.cid, time.time() - start_time))

    def train_DML(self, conf, device):
        """
        Def-KT中的模型相互学习训练
        """
        start_time = time.time()
        
        loss_fn, optimizers = self.pretrain_setup(conf, device)
        self.train_loss = []
        for i in range(conf.local_epoch): # 注意！！conf.local_epoch按论文说的设为了1
            batch_loss = []
            epoch_acc = []
            
            # 以下用了一个mini-batch的数据做了两个模型（远程模型A与本地模型B）的DML
            for batched_x, batched_y in self.train_loader:
                x, y = batched_x.to(device), batched_y.to(device)
                x, y = Variable(x), Variable(y)
                
                public_out = self.model(x) # self.model是从server下载的模型（或者说从A组客户端传过来的模型）
                local_out = self.local_model(x) # self.local_model是本地的模型
                
                # train local_model
                local_ce_loss= self.loss_ce(local_out, y) 
                # 交叉熵，y为真实标签（本地数据）
                local_kl_loss = self.loss_kl(F.log_softmax(local_out, dim = 1), 
                                            F.softmax(Variable(public_out), dim=1))
                # KL散度，public_out为公共模型（来自A组的模型）的输出，local_out为本地模型的输出
                
                local_loss = local_ce_loss + 0.3 * local_kl_loss
                # 计算总的loss，给kl_loss一个较小的权重，确保交叉熵的主导地位，从而保证模型能往正确的方向学习
                # Deep Mutual Learning 论文中有讨论
                local_loss.requires_grad_(True)

                optimizers[1].zero_grad()
                local_loss.backward()
                optimizers[1].step()

                # train public_model
                # 因为是相互学习，两个模型都要进行一样的训练
                public_ce_loss= self.loss_ce(public_out, y)
                public_kl_loss = self.loss_kl(F.log_softmax(public_out, dim = 1), 
                                            F.softmax(Variable(local_out), dim = 1))
                
                public_loss = public_ce_loss + 0.5 * public_kl_loss # 权重为啥要这么设置？

                optimizers[0].zero_grad()
                public_loss.backward()
                optimizers[0].step()

                # 最后将更新后的模型A的loss记入本次batch的loss列表中
                batch_loss.append(public_loss.item())
                epoch_acc.append(accuracy(public_out, y)) # 算一下准确率
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            # 将模型A的平均batch_loss记为本轮epoch的loss
            print('Client {} DML epoch: {}, loss: {:.4f}'.format(self.cid, i, current_epoch_loss))
            self.train_loss.append(float(current_epoch_loss))
            # 本轮epoch的loss加入到总的loss列表中
            torch.cuda.empty_cache()
        self.train_time = time.time() - start_time
        self.local_model = self.model # 将本地模型更新为模型A

    # train 方法每个round调用一次
    def train(self, conf, device, train_local_only):
        """
        每个round中
        1. A组客户端调用一次train，train_local_only为True，单独训练本地模型
        2. B组客户端调用一次train，train_local_only为False，与server分发过来的train过的A组模型进行DML
        """
        if train_local_only:
            self.train_local_only(conf, device)
        else:
            self.train_DML(conf, device)

    # 因为重写后的train的参数列表与默认不同，所以需要连run_train一起重写
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


    def load_optimizer(self, conf):
        optimizers=[]
        optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=conf.optimizer.lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay,
                                        nesterov=conf.optimizer.nesterov)
        optimizers.append(optimizer)
        local_optimizer = torch.optim.SGD(self.local_model.parameters(),
                                        lr=conf.optimizer.lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay,
                                        nesterov=conf.optimizer.nesterov)
        # local_optimizer1=copy.deepcopy(local_optimizer)
        optimizers.append(local_optimizer)
        return optimizers
    
    def pretrain_setup(self, conf, device):
        """Setup loss function and optimizer before training."""
        self.simulate_straggler()
        self.model.train()
        self.model.to(device)
        self.local_model.train()
        self.local_model.to(device)
        loss_fn = self.load_loss_fn(conf)
        optimizers = self.load_optimizer(conf)
        if self.train_loader is None:
            self.train_loader = self.load_loader(conf)
        return loss_fn, optimizers
    
