"""
    Some handy functions for pytroch model training ...
"""
import copy
import logging
import os
import random

import psutil
import torch


def get_model(config):
    """
    Get the model by the alias.
    You should add the models' names here to load
    :param config: the input arguments
    :return: the model
    """
    alias = config.alias

    trainer = None

    if alias == 'FedMAE':
        from model import model
        trainer = model.FedMAETrainer(config)
        config.is_federated = True
    else:
        raise ValueError('unknown model name: ' + alias)

    return trainer


def get_dataloader(config):
    from utils.data import dataset

    data = dataset.Dataset(config)

    if 'VAE' in config.alias or 'MAE' in config.alias:
        from utils.data import ae
        loader = ae.AEDataLoader(config, data)
    else:
        from utils.data import cf
        loader = cf.CFDataLoader(config, data)

        if config.alias == 'LightGCN':
            config.adj_mat = data.adj_mat

    return loader


# Checkpoints
def saveCheckPoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resumeCheckPoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(
                                device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def sampleClients(client_list, sample_strategy='random', sample_ratio=0.1, last_clients=None):
    """
    Sample clients from the client list.
    :param client_list: a list of client ids
    :param sample_strategy: str, the sample strategy
    :param sample_ratio: float, the sample ratio
    :param last_clients: a list of client ids
    :return: a list of client ids
    """

    if sample_ratio == 1:
        # Use all clients
        participants = client_list
    else:
        # Calculate the number of clients to be sampled
        sample_num = int(len(client_list) * sample_ratio)

        # Remove the clients that have been sampled in the last round
        if last_clients is not None:
            client_list = list(set(client_list) - set(last_clients))

        if sample_strategy == 'random':
            # Randomly sample clients
            participants = random.sample(client_list, sample_num)
        else:
            raise ValueError('Invalid sample strategy: {}'.format(sample_strategy))

    return participants


def initLogging(log_file_name):
    """Initialize logging configuration."""
    import logging
    import coloredlogs

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file_name,
        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    coloredlogs.install(
        level='DEBUG',
        fmt='%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def setSeed(seed=0):
    """Set all random seeds"""

    import random
    import numpy as np
    import torch

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mail_notice(config):
    """
    Email notice that the training is finished.
    :param config: the input arguments
    :return: None
    """
    import iMail
    from utils import constant as uc
    import yaml

    logging.info('>' * 3 + ' Sending an email to notice that the training is finished... ')

    # Set the Mail Sender
    mail_obj = iMail.EMAIL(host=uc.EMAIL_HOST, sender_addr=uc.SENDER_ADDRESS, pwd=uc.PASSWORD,
                           sender_name=uc.SENDER_NAME)
    mail_obj.set_receiver(uc.RECEIVER)

    # Create a new email
    mail_title = '[NOTICE FROM EXPERIMENT] {} on {}/{}: {}-{}'.format(
        config.alias, config.dataset.upper(), config.data_file.split('.')[0].upper(),
        config.type, config.comment
    )
    mail_obj.new_mail(subject=mail_title, encoding='UTF-8')

    # Attach a text and a log file to the receiver
    file = config.log_file_name
    content = None
    with open(file, 'r', encoding='utf-8') as f:

        # Add the text
        for line in f:
            if '[Test] Best HR' in line:
                content = line.split('[Test]')[-1]
                mail_obj.add_text(content=yaml.dump(content))
                break

    mail_obj.attach_files(file)

    # Send the email
    mail_obj.send_mail()

    return mail_title, content


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Convert bytes to GB


def get_total_available_memory():
    mem = psutil.virtual_memory()
    return mem.available / (1024 ** 3)  # Convert bytes to GB


def copy_model_with_grad(model):
    model_copy = copy.deepcopy(model)
    for param, param_copy in zip(model.parameters(), model_copy.parameters()):
        if param.grad is not None:
            param_copy.grad = param.grad.clone()
    return model_copy
