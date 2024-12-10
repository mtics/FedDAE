import copy
import datetime
import logging
import os
import tracemalloc

import numpy as np
import pandas as pd

from abc import *
from utils import evaluation, utils
from utils.data import ae


class BaseTrainer(metaclass=ABCMeta):

    def __init__(self, args):
        self.args = args
        self.metric = evaluation.Metric(args)
        self.fabric = args.fabric
        self.optimizer = None

    def train(self, data_loader):

        train_set, val_set, test_set = data_loader.get_dataloaders()

        losses, times = [], []
        hrs, ndcgs = {'test': [], 'val': []}, {'test': [], 'val': []}
        best_val_ndcg, test_iter = 0, 0

        logging.info('--------------- The model training is started ---------------')

        available_memory = utils.get_total_available_memory()
        init_memory = utils.get_memory_usage()

        for iteration in range(self.args.num_iterations):

            # 1. Train for one iteration
            start = datetime.datetime.now()
            train_loss = self._train_one_iter(train_set, iteration)
            train_memory = utils.get_memory_usage()
            end = datetime.datetime.now()
            elapsed = (end - start).total_seconds()
            times.append(elapsed)

            losses.append(train_loss)

            logging.info('[{}][{}/{}][Epoch {}/{}][Train] Memory Usage = {:.4f} GB / {:.4f} GB'.format(
                self.args.alias, self.args.dataset, self.args.data_file.split('.')[0],
                iteration + 1, self.args.num_iterations, train_memory - init_memory, available_memory))

            logging.info('[Epoch {}/{}][Train] Loss = {:.4f}, Time consuming: {:.2f}s'.format(
                iteration + 1, self.args.num_iterations, train_loss, elapsed))

            # 2. Evaluate HR@10 and NDCG@10 on the validation set
            val_hr, val_ndcg = self.evaluate(val_set)

            hrs['val'].append(val_hr)
            ndcgs['val'].append(val_ndcg)

            logging.info('[Epoch {}/{}][Valid] HR@{} = {:.4f}, NDCG@{} = {:.4f}'.format(
                iteration + 1, self.args.num_iterations, self.args.top_k, val_hr, self.args.top_k, val_ndcg))

            # Choose the best model based on the validation NDCG@10
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                test_iter = iteration

            # 3. Evaluate HR@10 and NDCG@10 on the test set
            hr, ndcg = self.evaluate(test_set)

            logging.info('[Epoch {}/{}][Test]  HR@{} = {:.4f}, NDCG@{} = {:.4f}'.format(
                iteration + 1, self.args.num_iterations, self.args.top_k, hr, self.args.top_k, ndcg))

            hrs['test'].append(hr)
            ndcgs['test'].append(ndcg)

            if self.args.early_stop and iteration > 1 and \
                    abs(losses[-1] - losses[-2]) / (losses[-1] + 1e-6) < self.args.tol:
                logging.info('[{}][{}/{}] Early stopping at the {}-th iteration'.format(
                    self.args.alias, self.args.dataset, self.args.data_file.split('.')[0], iteration + 1))
                break

        logging.info('--------------- The model training is finished ---------------')

        self.args.model_dir = self.args.model_dir.format(self.args.alias, self.args.dataset, self.args.type, iteration)

        logging.info('HR@10 list: {}'.format([format(_, '.4f') for _ in hrs['test']]).replace("'", ""))
        logging.info('NDCG@10 list: {}'.format([format(_, '.4f') for _ in ndcgs['test']]).replace("'", ""))
        logging.info('Losses: {}'.format([format(_, '.2f') for _ in losses]).replace("'", ""))

        logging.info('[{}/{}][{}] Time consuming: {:.2f}s'.format(
            self.args.dataset, self.args.data_file.split('.')[0], self.args.alias, sum(times)))

        notice = '[Test] Best HR: {:.4f}, NDCG: {:.4f} at the {}-th iteration'.format(
            hrs['test'][test_iter], ndcgs['test'][test_iter], test_iter + 1)

        logging.info(notice)

        self._save_results(hrs, ndcgs, losses, test_iter)

    def _save_results(self, hrs, ndcgs, losses, test_iter):

        logging.info('[Saving] All results are being saved ...')

        config = copy.deepcopy(self.args.__dict__)

        # delete some unuseful key-value
        useless_keys = [
            'device_id', 'hardware', 'num_gpus', 'on_server', 'model_dir', 'fabric', 'paths',
            'num_iterations', 'data_path', 'early_stop', 'batch_size', 'is_federated', 'log_file_name'
        ]
        for key in useless_keys:
            del config[key]

        logging.info(str(config))

        # save useful data
        result_file = self.args.paths['save'] + '[{}.{}].{}.txt'.format(
            config['dataset'], config['data_file'].split('.')[0], config['type'])

        with open(result_file, 'a') as file:
            file.write(str(config) + '\n')

        data_file = self.args.paths['save'] + '[{}]-[{:.2e}-{:.2e}]-[HR{:.4f}-NDCG{:.4f}]-[{}.{}].npz'.format(
            config['data_file'].split('.')[0], config['beta'], config['gamma'],
            hrs['test'][test_iter], ndcgs['test'][test_iter], config['type'], config['comment']
        )

        np.savez(data_file, hrs=hrs, ndcgs=ndcgs, losses=losses)

        # Handle performances in the specified csv files
        csv_name = os.path.join(self.args.paths['save'],
                                '{}-performances.csv'.format(config['data_file'].split('.')[0]))
        performances = {
            'Type': self.args.type,
            'Comment': self.args.comment,
            'HR': hrs['test'][test_iter],
            'NDCG': ndcgs['test'][test_iter],
            'MEAN-HR': 0,
            'STD-HR': 0,
            'MEAN-NDCG': 0,
            'STD-NDCG': 0,
        }

        # calculate mean & std
        if os.path.exists(csv_name):
            csv_df = pd.read_csv(csv_name)
            pos = (csv_df['Type'] == config['type']) & (csv_df['Comment'] == config['comment'])

            if pos.any():
                idx = pos.index[0]
            else:
                idx = len(csv_df)

            csv_df.loc[idx] = performances
            csv_df['MEAN-HR'] = csv_df['HR'].mean()
            csv_df['STD-HR'] = csv_df['HR'].std()
            csv_df['MEAN-NDCG'] = csv_df['NDCG'].mean()
            csv_df['STD-NDCG'] = csv_df['NDCG'].std()
        else:
            for key in performances.keys():
                performances[key] = [performances[key]]
            csv_df = pd.DataFrame(performances)

        # write performances into csv
        csv_df.to_csv(csv_name, index=False)

        logging.info('[Finished] All results have been saved successfully!')

        if self.args.notice:
            mail_title, mail_content = utils.mail_notice(self.args)
            if self.args.on_server:
                utils.autodl_notice(mail_title, mail_content)

    def set_optimizer(self, *args, **kwargs):
        pass

    def _train_one_iter(self, train_loader, iteration):
        pass

    def evaluate(self, eval_data):
        pass

    def calculate_loss(self, *args, **kwargs):
        pass

    def _update_hyperparams(self, *args, **kwargs):
        pass


class CentralTrainer(BaseTrainer):

    def __init__(self, args):
        super(CentralTrainer, self).__init__(args)

    def _train_one_iter(self, train_loader, iteration):
        assert hasattr(self, 'model'), 'Please specify the model'

        self.model.train()

        total_loss = 0
        for idx, batch in enumerate(train_loader):
            total_loss += self._train_one_batch(batch)

        # Update the model hyperparameters
        self._update_hyperparams(iteration)

        return total_loss

    @abstractmethod
    def _train_one_batch(self, batch, *args, **kwargs):
        pass


class FederatedTrainer(BaseTrainer):

    def __init__(self, args):
        super(FederatedTrainer, self).__init__(args)

        self.global_model = {}
        self.client_models = {}
        self.optimizers = {}

        self.last_participants = None

    def _train_one_iter(self, train_loader, iteration):
        assert hasattr(self, 'model'), 'Please specify the model'

        # Randomly select a subset of clients
        sampled_clients = utils.sampleClients(
            list(train_loader.keys()), self.args.clients_sample_strategy,
            self.args.clients_sample_ratio, self.last_participants
        )

        # Store the selected clients for the next round
        self.last_participants = sampled_clients

        participant_params = {}
        train_loss = 0
        for user in sampled_clients:

            client_loader = train_loader[user]
            client_model, client_optimizer = self._set_client(user, iteration)

            client_model.train()

            client_losses = []
            # Train the client model
            for epoch in range(self.args.local_epoch):

                client_loss = 0
                for idx, batch in enumerate(client_loader):
                    client_model, client_optimizer, loss = self._train_one_batch(batch, client_model, client_optimizer)
                    client_loss += loss

                client_losses.append(client_loss / len(client_loader))

                if epoch > 0 and abs(client_losses[-1] - client_losses[-2]) / (
                        client_losses[-1] + 1e-6) < self.args.tol:
                    break

            train_loss += client_losses[-1]

            # Store the client model parameters need to be aggregated
            participant_params[user] = self._store_client_model(user, client_model)

        # Aggregate the client model parameters in the server side
        self._aggregate_params(participant_params)

        # Update the model hyperparameters
        self._update_hyperparams(iteration)

        return train_loss

    def _set_client(self, *args, **kwargs):
        pass

    def _train_one_batch(self, batch, *args, **kwargs):
        pass

    def _aggregate_params(self, *args, **kwargs):
        pass

    def _store_client_model(self, *args, **kwargs):
        pass
