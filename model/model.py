import copy
import math

import torch
import torch.nn.functional as F

from utils.vae import MAE
from utils.trainer import FederatedTrainer
from utils import evaluation, utils


class FedMAETrainer(FederatedTrainer):

    def __init__(self, args):
        self.model = MAE(args)
        super(FedMAETrainer, self).__init__(args)
        print(self.model)

        self.gm = copy.deepcopy(self.model)

        self.beta = 1
        self.lr = args.lr
        self.client_lrs = {}

        self.optimizer = torch.optim.Adam(self.gm.parameters(), lr=self.lr, weight_decay=self.args.l2_reg)
        self.gm, self.optimizer = self.fabric.setup(self.gm, self.optimizer)

    def _train_one_iter(self, train_loader, iteration):
        # Randomly select a subset of clients
        sampled_clients = utils.sampleClients(
            list(train_loader.keys()), self.args.clients_sample_strategy,
            self.args.clients_sample_ratio, self.last_participants
        )

        # Store the selected clients for the next round
        self.last_participants = sampled_clients

        train_loss = 0

        self.gm.train()
        self.optimizer.zero_grad()
        for user in sampled_clients:
            client_loader = list(train_loader[user])
            assert len(client_loader) == 1, 'Only one client per user is allowed'

            client_model, client_optimizer = self._set_client(user, iteration)
            client_model, client_optimizer = self.fabric.setup(client_model, client_optimizer)

            client_model.train()

            client_loss = 0
            x = self.fabric.to_device(client_loader[0])
            for epoch in range(self.args.local_epoch):
                pred, mu, logvar, _ = client_model(x)
                loss = self.calculate_loss(pred, x, mu, logvar)
                self.fabric.backward(loss)

                for pA, pB in zip(client_model.parameters(), self.gm.parameters()):
                    pB.grad = pA.grad + (pB.grad if pB.grad is not None else 0)

                client_optimizer.step()

                client_loss += loss.item()

            train_loss += client_loss / len(client_loader)

            self.client_models[user] = {}
            param_dict = copy.deepcopy(client_model.to('cpu')).state_dict()
            for key in param_dict.keys():
                if 'ql' in key or 'gate' in key:
                    self.client_models[user][key] = param_dict[key]

            for param_group in client_optimizer.param_groups:
                self.client_lrs[user] = param_group['lr']

        for p in self.gm.parameters():
            p.grad /= len(sampled_clients)

        self.optimizer.step()

        self._update_hyperparams(iteration)

        return train_loss / len(sampled_clients)

    def _set_client(self, *args, **kwargs):
        user, iteration = args

        client_model = copy.deepcopy(self.model)
        if user in self.client_models.keys() and iteration != 0:
            client_model.decoder.load_state_dict(self.gm.decoder.state_dict())
            client_model.qg.load_state_dict(self.gm.qg.state_dict())
            client_model.ql.load_state_dict(self.gm.ql.state_dict())
            for key in self.client_models[user].keys():
                client_model.state_dict()[key] = copy.deepcopy(self.client_models[user][key])

        client_optimizer = torch.optim.Adam(client_model.parameters(),
                                            lr=self.client_lrs[user] if user in self.client_lrs.keys() else self.lr,
                                            weight_decay=self.args.l2_reg)

        return client_model, client_optimizer

    def _update_hyperparams(self, *args, **kwargs):
        iteration = args[0]
        self.beta = math.tanh(iteration / 10) * self.beta

        # self.lr = self.args.lr * math.exp(- self.args.decay_rate * iteration)
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = self.lr

    def calculate_loss(self, *args, **kwargs):
        pred, truth, mu, logvar = args[0], args[1], args[2], args[3]

        BCE = -(F.log_softmax(pred, 1) * truth).sum(1).mean()

        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        loss = BCE + self.beta * KLD

        return loss

    @torch.no_grad()
    def evaluate(self, eval_data):

        all_pred, all_truth = [], []

        self.gm.eval()
        for user, loader in eval_data.items():
            client_model, _ = self._set_client(user, 1)
            client_model = self.fabric.to_device(client_model)

            client_model.eval()

            client_metrics = None
            batch = list(eval_data[user])[0]
            batch = self.fabric.to_device(batch)

            x, truth = batch

            pred, _, _, _ = client_model(x)
            pred[x == 1] = -float("Inf")

            all_pred.append(pred)
            all_truth.append(truth)

        tensor_pred = torch.cat(all_pred, dim=0)
        tensor_truth = torch.cat(all_truth, dim=0)

        metrics = evaluation.recalls_and_ndcgs_for_ks(tensor_pred, tensor_truth)

        # print(metrics)
        return metrics['Recall@{}'.format(self.args.top_k)], metrics['NDCG@{}'.format(self.args.top_k)]
