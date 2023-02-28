import easyfl

# Initialize federated learning with default configurations.
config_file = "config.yaml"
# Load and combine these two configs.
config = easyfl.load_config(config_file)
easyfl.init(config)
# Execute federated learning training.
easyfl.run()