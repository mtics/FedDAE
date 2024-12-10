import datetime
import datetime
import os

import torch

from utils import utils, parser
from utils.data import dataset, ae
from lightning.fabric import Fabric

if __name__ == '__main__':
    # Config
    config = parser.loadParser()

    # Set GPU
    fabric = Fabric(accelerator=config.hardware, devices=[config.device_id])
    fabric.launch()
    config.fabric = fabric

    # Set random seed
    utils.setSeed(config.seed)

    # Check if folders exist
    config.paths = {
        'log': 'outputs/logs/{}/{}/'.format(config.alias, config.dataset),
        'checkpoint': 'outputs/checkpoints/{}/{}/'.format(config.alias, config.dataset),
        'save': 'outputs/results/{}/{}/{}/'.format(config.alias, config.dataset, config.type),
    }

    for p in config.paths.keys():
        if not os.path.exists(config.paths[p]):
            os.makedirs(config.paths[p])

    config.model_dir = os.path.join(config.paths['checkpoint'], '[{}.{}]{}.Epoch{}.ckpt')

    # Logging.
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    log_file_name = os.path.join(config.paths['log'], '[{}]-[{}.{}]-[{}.{}]-[{}].txt'.format(
        config.alias, config.dataset, config.data_file.split('.')[0], config.type, config.comment, current_time))

    config.log_file_name = log_file_name

    utils.initLogging(log_file_name)

    # Load data
    loader = utils.get_dataloader(config)

    # Load model
    model = utils.get_model(config)

    # Train
    model.train(loader)

    if config.alias == 'FedMAE':
        state = {
            'global_model': model.gm,
            'client_models': model.client_models
        }
        # Save model
        fabric.save(config.model_dir, state)

    loader
