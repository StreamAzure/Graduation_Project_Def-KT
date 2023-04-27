import easyfl
from CustomizedClient import CustomizedClient
from CustomizeServer  import CustomizedServer
from CustomizedCNN import CustomizedCNN

# Register the customized model class.
easyfl.register_model(CustomizedCNN)

# Register customized client.
easyfl.register_server(CustomizedServer)
easyfl.register_client(CustomizedClient)

# Initialize federated learning with default configurations.
config_file = "config.yaml"
# Load and combine these two configs.
config = easyfl.load_config(config_file)
easyfl.init(config)
# Execute federated learning training.
easyfl.run()