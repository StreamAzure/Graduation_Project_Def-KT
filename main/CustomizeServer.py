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
        self.modellist=[]
        pass  # more initialization of attributes.
    
    def aggregation(self):
        uploaded_content = self.get_client_uploads()
        self.modellist = list(uploaded_content[MODEL].values())
        random.shuffle(self.modellist)
        # Original implementation of aggregation weights
        # weights = list(uploaded_content[DATA_SIZE].values())
        # We can assign the manipulated customized weights in aggregation.   
        # customized_weights = list(range(len(models)))
        # model = self.aggregate(models, customized_weights)
        # self.set_model(model, load_dict=True)
       
        vaild_clients=  random.sample(list(set(self._clients)-set(self.selected_clients)), len(self.selected_clients))
        vaild_votes=[]
        weights=[]
        for c in vaild_clients:
            vaild_votes.append(c.run_vaildmodel(self.modellist,self.conf.device))
        for i in range(len(self.modellist)):
            weights.append(0.5+vaild_votes.count(i)/len(self.modellist))

        print(vaild_votes)
        model = self.aggregate(self.modellist, weights)
        self.set_model(model, load_dict=True)

        

            
    # 下面这段代码是各个客户端不聚合，直接送到下一个客户端进行训练，完全参照论文的算法Def-KT 

    # def distribution_to_train_locally(self):
    #     uploaded_models = {}
    #     uploaded_weights = {}
    #     uploaded_metrics = []
    #     for client in self.grouped_clients:
    #         # Update client config before training
    #         self.conf.client.task_id = self.conf.task_id
    #         self.conf.client.round_id = self._current_round
    #         if (len(self.modellist)==0):
    #             uploaded_request = client.run_train(self._compressed_model, self.conf.client)
    #         else:
    #             uploaded_request = client.run_train(self.modellist[int(client.cid[-3:])%(len(self.modellist)-1)], self.conf.client)
    #         uploaded_content = uploaded_request.content

    #         model = self.decompression(codec.unmarshal(uploaded_content.data))
    #         uploaded_models[client.cid] = model
    #         uploaded_weights[client.cid] = uploaded_content.data_size
    #         uploaded_metrics.append(metric.ClientMetric.from_proto(uploaded_content.metric))

    #     self.set_client_uploads_train(uploaded_models, uploaded_weights, uploaded_metrics)
    

