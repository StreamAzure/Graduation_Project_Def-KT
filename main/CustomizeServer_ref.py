from itertools import count
import easyfl
from easyfl.server import BaseServer
from easyfl.server.base import MODEL
from easyfl.protocol import codec
from easyfl.tracking import metric
from easyfl.server import strategies
import random

class CustomizedServer(BaseServer):

    def __init__(self, conf, **kwargs):
        super(CustomizedServer, self).__init__(conf, **kwargs)
        self.modellist=[] # 自定义的modellist，用于存储每一轮的客户端模型
        pass  # more initialization of attributes.

    # 下面这段代码是各个客户端不聚合，直接送到下一个客户端进行训练，完全参照论文的算法Def-KT 

    def distribution_to_train_locally(self):
        uploaded_models = {}
        uploaded_weights = {}
        uploaded_metrics = []
        for client in self.grouped_clients:
            # Update client config before training
            self.conf.client.task_id = self.conf.task_id
            self.conf.client.round_id = self._current_round

            # 默认：uploaded_request = client.run_train(self._compressed_model, self.conf.client)
            # 默认：使用服务器分发的模型进行训练
            
            ### 自定义部分
            if (len(self.modellist)==0): # 如果modellist为空，即第一轮，直接训练服务器分发的模型
                uploaded_request = client.run_train(self._compressed_model, self.conf.client)
            else:
                uploaded_request = client.run_train(self.modellist[int(client.cid[-3:])%(len(self.modellist)-1)],
                                                     self.conf.client)
            # 否则，每个客户端从modellist中选出一个模型进行训练，选择方法是取cid的后三位数对modellist的长度取余
            # modellist是上一轮各个客户端的模型
            # 例如，modellist长度为5，cid为client_001，那么取cid的后三位数对modellist的长度取余，即001%5=1
            
            ### 自定义部分

            uploaded_content = uploaded_request.content

            model = self.decompression(codec.unmarshal(uploaded_content.data))
            uploaded_models[client.cid] = model
            uploaded_weights[client.cid] = uploaded_content.data_size
            uploaded_metrics.append(metric.ClientMetric.from_proto(uploaded_content.metric))

        self.set_client_uploads_train(uploaded_models, uploaded_weights, uploaded_metrics)
    
    def aggregation(self):
        uploaded_content = self.get_client_uploads() 
        # 获取客户端上传模型 
        self.modellist = list(uploaded_content[MODEL].values()) 
        # 将全部客户端上传模型存入modellist
        # uploaded_content[MODEL]是一个字典，键是客户端id，值是客户端上传的模型
        # uploaded_content[DATA_SIZE]是一个字典，键是客户端id，值是客户端上传的模型的大小

        # 在默认Server中（FedAvg方式），weights = list(uploaded_content[DATA_SIZE].values())
        # 即各个模型的权重与模型大小相同
        # 而下面用不同的权重计算方式重新定义了weights

        random.shuffle(self.modellist) # 将modellist中的模型随机打乱
       
        vaild_clients=  random.sample(list(set(self._clients)-set(self.selected_clients)), 
                                      len(self.selected_clients))
        # 从未选中的客户端（已去重）中随机挑选len(self.selected_clients)个客户端
        vaild_votes=[]

        weights=[]
        for c in vaild_clients:
            vaild_votes.append(c.run_vaildmodel(self.modellist,self.conf.device)) 
            # 对于每一个客户端，从modellist中选出对它而言Loss最小的model，下标加入到valid_votes中
        for i in range(len(self.modellist)):
            weights.append(0.5+vaild_votes.count(i)/len(self.modellist))
            # valid_votes.count(i)统计i出现的次数
            # 如共5个客户端，下标为2的模型出现次数为3，表明该模型在3个客户端中都是Loss最小的
            # 以这个出现次数与总模型数量的比值作为该模型的权重

        print(vaild_votes)
        model = self.aggregate(self.modellist, weights)
        self.set_model(model, load_dict=True)


