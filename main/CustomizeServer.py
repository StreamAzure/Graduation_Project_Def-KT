from easyfl.server import BaseServer
from easyfl.server.base import MODEL
from easyfl.protocol import codec
import numpy as np
import logging


"""
Def-KT论文复现
1. 论文要求每轮选出2Q个客户端，且2Q<<K，K为总客户端数。所以这里暂定Q=5，K=100
2. 每一轮开始时，A组客户端自行训练。训练完毕后，id为 j 的客户端将模型A传递给id为 (j+Q)%K 的B组客户端
3. B组客户端将模型A与自己的本地模型B进行相互学习（交叉熵+KL散度），最后将更新的模型A保存作为自己的本地模型
4. 
"""
logger = logging.getLogger(__name__)

K = 100 # 客户端总数
Q = 5 # 一组客户端数，每轮选两组
# 注：“每轮随机选择Q个”已由默认的Selection方法实现，
# 参数由config.yaml中的clients_per_round及random_selection决定

"""
Server 的默认 train 方法执行顺序：
1. self.selection()，选出指定的Q个客户端加入self.selected_clients
2. self.grouping_for_distributed()，用于GPU分组，只用1个GPU就不用管，self.grouped_clients = self.selected_clients
3. self.compression()，压缩模型，啥也没干
4. self.distribution_to_train()
    将模型分发给客户端，并调用客户端的run_train方法让客户端开始训练
    并自动收集训练完毕的模型，记在uploaded_models和uploaded_weights中
5. self.aggregation()，
    将客户端上传的模型聚合，默认聚合方法是FedAvg
    只做权重分配，实际聚合计算在self.aggregate()中
"""

class CustomizedServer(BaseServer):
    """
    server实际只需要做selection和distribution_to_train，不需要收集模型和聚合计算
    """

    def __init__(self, conf, **kwargs):
        super(CustomizedServer, self).__init__(conf, **kwargs)
        self.A_clients = [] # A组客户端
        self.B_clients = [] # B组客户端

    def selection(self, clients, clients_per_round):
        """
        似乎没办法在distribution_to_train中通过指定ID的形式把模型分发给指定的客户端
        所以只能在这里把A组和B组一起选出
        """
        if clients_per_round > len(clients):
            logger.warning("Available clients for selection are smaller than required clients for each round")

        clients_per_round = min(clients_per_round, len(clients))
        # 这里写死了一定是随机选择（默认版本是有选项的）
        # 选出Q个客户端加入self.A_clients
        np.random.seed(self._current_round)
        self.A_clients = np.random.choice(clients, clients_per_round, replace=False)
        self.A_clients = self.A_clients.tolist()

        # 再从余下的客户端中选出Q个客户端加入self.B_clients
        # 论文所说的B组由下标计算确定，感觉没法完全保证A组和B组的客户端不重复
        # 所以这里换了种方法，直接从剩下的客户端中随机选出Q个
        self.B_clients = np.random.choice(list(set(clients) - set(self.A_clients)), 
                                          clients_per_round, replace=False)
        self.B_clients = self.B_clients.tolist()

    
    def distribution_to_train_locally(self):
        """
        1. 让A组客户端进行本地训练
        2. 把训练好的A组客户端模型分发给B组客户端进行DML
        """
        A_models = []
        for client in self.A_clients: # 遍历已选出的Q个A组客户端中的每一个
            self.conf.client.task_id = self.conf.task_id
            self.conf.client.round_id = self._current_round

            print("client: {}".format(client.cid))

            uploaded_request = client.run_train(self._compressed_model, self.conf.client, train_local_only=True)
            # A组客户端训练自己的本地模型，分发的self.model实际上不会被用到（应该，吧？）
            model = self.decompression(codec.unmarshal(uploaded_request.content.data))
            # 把A组客户端训练好的模型保存为model
            A_models.append(model)

        for client in self.B_clients: # 遍历已选出的Q个B组客户端中的每一个
            # 将A组model分发给对应的B组客户端进行DML训练
            model = A_models[self.B_clients.index(client)]
            client.run_train(model, self.conf.client, train_local_only=False)
            
        
    
    def aggregation(self):
        pass # 不对收集上来的客户端模型做任何聚合

    def test(self):
        pass


