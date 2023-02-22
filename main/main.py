import easyfl
from CustomizedClient import CustomizedClient
from CustomizeServer  import CustomizedServer

# config = {
#     "data": {"dataset": "femnist", "num_of_clients": 10},
#     "server": {"rounds": 50, "clients_per_round": 5, "test_all": False},
#     "client": {"local_epoch": 10},
#     "model": "simple_cnn",
#     "gpu":1,
#     "test_mode":"test_in_client"
# }



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