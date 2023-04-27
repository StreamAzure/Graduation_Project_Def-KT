from torch import nn
import torch.nn.functional as F
import easyfl
from easyfl.models import BaseModel
from CustomizedCNN import CustomizedCNN

# Register the customized model class.
easyfl.register_model(CustomizedCNN)

config_file = "config.yaml"
config = easyfl.load_config(config_file)
easyfl.init(config)
easyfl.run()