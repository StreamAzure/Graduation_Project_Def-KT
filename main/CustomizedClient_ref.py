import easyfl
import time
from torch import nn

from easyfl.client.base import BaseClient
import  torch
import torch.nn.functional as F
import copy
from torch.autograd import Variable
from utils import accuracy
from local_cnn import CNNModel
from local_resnet import ResModel
import sys

# Inherit BaseClient to implement customized client operations.
class CustomizedClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(CustomizedClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        # Initialize a classifier for each client.
        self.local_model= None
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()
        self.local_model= CNNModel()

    def download(self, model):
        """Download model from the server.

        Args:
            model (nn.Module): Global model distributed from the server.
        """
        if self.compressed_model:
            self.compressed_model.load_state_dict(model.state_dict())
        else:
            self.compressed_model = copy.deepcopy(model)
            

    # def decompression(self):

    #     if(self.model==None):
    #         self.model = self.compressed_model
    #         return
    #     #仅加载两层线性层
    #     para_state_dict =self.compressed_model.state_dict()
    #     model_dict = self.model.state_dict()
    #     pretrained_dict = {k: v for k, v in para_state_dict.items() if k in model_dict and  not "fc2" in k}
    #     model_dict.update(pretrained_dict)
    #     self.model.load_state_dict(model_dict)

    # def compression(self):
    #     #仅加载两层线性层
    #     para_state_dict =self.model.state_dict()
    #     model_dict = self.compressed_model.state_dict()
    #     pretrained_dict = {k: v for k, v in para_state_dict.items() if k in model_dict and not "fc2" in k}
    #     model_dict.update(pretrained_dict)
    #     self.compressed_model.load_state_dict(model_dict)

    def train(self, conf, device):
        start_time = time.time()
        
        loss_fn, optimizers = self.pretrain_setup(conf, device)
        self.train_loss = []
        for i in range(conf.local_epoch):
            batch_loss = []
            epoch_acc=[]
            
            for batched_x, batched_y in self.train_loader:
                x, y = batched_x.to(device), batched_y.to(device)
                x, y = Variable(x), Variable(y)
                
                public_out = self.model(x) # self.model是从server下载的模型（或者说从A组客户端传过来的模型）
                local_out= self.local_model(x) # self.local_model是本地的模型
                
                # train local_model
                local_ce_loss= self.loss_ce(local_out, y)
                local_kl_loss = self.loss_kl(F.log_softmax(local_out, dim = 1), 
                                            F.softmax(Variable(public_out), dim=1))
                local_loss=local_ce_loss+0.3*local_kl_loss
                local_loss.requires_grad_(True)

                optimizers[1].zero_grad()
                local_loss.backward()
                optimizers[1].step()

                # train public_model
                public_ce_loss= self.loss_ce(public_out, y)
                public_kl_loss = self.loss_kl(F.log_softmax(public_out, dim = 1), 
                                            F.softmax(Variable(local_out), dim=1))
                
                public_loss=public_ce_loss+0.5*public_kl_loss
                
                optimizers[0].zero_grad()
                public_loss.backward()
                optimizers[0].step()

                batch_loss.append(public_loss.item())
            # print([x.grad for x in optimizers[1].param_groups[0]['params']])
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            # predicted = torch.max(local_out.data,1)[1]
            # correct += (predicted == batched_y).sum()
            self.train_loss.append(float(current_epoch_loss))
            torch.cuda.empty_cache()
        self.train_time = time.time() - start_time
               

    # A customized optimizer that sets different learning rates for different model parts.
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

    def post_train(self):
        """测试本地模型"""
        """testing local_model after testing public_model """
        self.local_model.eval()
        self.local_model.to(self.device)
        
        local_test_loss = 0
        correct = 0
        with torch.no_grad():
            for batched_x, batched_y in self.test_loader:
                x = batched_x.to(self.device)
                y = batched_y.to(self.device)
                public_out = self.local_model(x)
                log_probs = self.local_model(x)
                local_ce_loss= self.loss_ce(log_probs, y)
                local_kl_loss = self.loss_kl(F.log_softmax(log_probs, dim = 1), 
                                                        F.softmax(public_out, dim=1))
                local_loss=local_ce_loss+local_kl_loss
                _, y_pred = torch.max(log_probs, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                local_test_loss += local_loss.item()
            test_size = self.test_data.size(self.cid)
            local_test_loss /= test_size
            local_test_accuracy = 100.0 * float(correct) / test_size
            s='Client {}, testing -- Local Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            self.cid, local_test_loss, correct, test_size, local_test_accuracy)
            print(s)
            # with open('local_acc.log','wb+') as f: # 输出各个客户端的测试结果到文件
            #    f.write(s) 

            # 检测各个子类准确率
            # total_num = 0
            # t_correct = list(0. for i in range(10))
            # t_total = list(0. for i in range(10))
            # for images, labels in self.train_loader:
            #     images = images.to(self.device)
            #     labels = labels.to(self.device)

            #     output = self.model(images)

            #     prediction = torch.argmax(output, 1)
            #     res = prediction == labels
            #     for label_idx in range(len(labels)):
            #         label_single = labels[label_idx]
            #         t_correct[label_single] += res[label_idx].item()
            #         t_total[label_single] += 1
            # for i in range(10):
            #     if(t_total[i]==0):
            #         print('Accuracy of %5s : %2d %%' % ( i, 0))
            #     else:
            #         print('Accuracy of %5s : %2d %%' % (
            #             i, 100 * t_correct[i] / t_total[i]))


    def run_vaildmodel(self,modellist,device):
        
        loss_fn = self.load_loss_fn(self.conf)
        val_list=[]
        if self.train_loader is None:
            self.train_loader = self.load_loader(self.conf)
        vote=0
        vote_val=sys.float_info.max
        for index,model in enumerate(modellist):
            model.eval()
            model.to(device)
            size=0
            val_loss=0
            val_acc=0.0
            correct=0
            with torch.no_grad():
                for batched_x, batched_y in  self.train_loader:
                    size=size+self.train_loader.batch_size
                    x = batched_x.to(device)
                    y = batched_y.to(device)
                    out=model(x)
                    _, y_pred = torch.max(out, -1)
                    loss = loss_fn(out, y)
                    correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                    val_loss+=loss.item()
                    if(size>=self.test_data.size(self.cid)):
                        break
                val_loss /= size
                val_acc= 100*float(correct) / size
                # val_list.append([val_loss,val_acc])
                if (vote_val>val_loss):
                    vote_val=val_loss
                    vote=index
        return vote