import random

from .abstractClass import AbstractDataLoader
import torch.utils.data as data_utils
import torch


class CFTrainDataset(data_utils.Dataset):

    def __init__(self, positives, negatives):
        # Row indices for sparse matrix
        #   e.g. [0, 0, 0, 1, 1, 4, 4, 4, 4]
        #        when user2items = {0:[1,2,3], 1:[4,5], 4:[6,7,8,9]}

        users, interact_items, ratings = [], [], []
        for user, items in positives.items():
            single_user, user_items, user_ratings = [], [], []
            for item in items:
                single_user.append(int(user))
                user_items.append(int(item))
                user_ratings.append(1.)
            for item in negatives[user]:
                single_user.append(int(user))
                user_items.append(int(item))
                user_ratings.append(0.)

            assert len(single_user) == len(user_items) == len(user_ratings)
            users.extend(single_user)
            interact_items.extend(user_items)
            ratings.extend(user_ratings)

        self.user_tensor = torch.LongTensor(users)
        self.item_tensor = torch.LongTensor(interact_items)
        self.rating_tensor = torch.FloatTensor(ratings)

    def __len__(self):
        return self.user_tensor.size(0)

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.rating_tensor[index]


class CFEvalDataset(data_utils.Dataset):

    def __init__(self, positives, train_positives):
        users, interact_items, train_items = [], [], []
        for user, items in positives.items():
            single_user, user_items, user_train_items = [], [], []
            for item in items:
                single_user.append(int(user))
                user_items.append(int(item))
            for item in train_positives[user]:
                user_train_items.append(int(item))

            assert len(single_user) == len(user_items)
            users.extend(single_user)
            interact_items.extend(user_items)
            train_items.extend(user_train_items)

        self.user_tensor = torch.LongTensor(users)
        self.item_tensor = torch.LongTensor(interact_items)
        self.train_tensor = torch.LongTensor(train_items)

    def __len__(self):
        return self.user_tensor.size(0)

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.train_tensor


class CFDataLoader(AbstractDataLoader):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        negatives = dataset.negatives
        negatives['negatives'] = negatives['negative_items'].apply(
            lambda x: random.sample(x, args.num_negative)
        )
        self.negatives = dict(negatives.groupby('userId').apply(lambda d: list(d['negatives'])))
        self.negatives = {k: v[0] for k, v in self.negatives.items()}

        self.data = {
            'train': self.train,
            'val': self.val,
            'test': self.test
        }

        negatives

    def get_dataloaders(self):
        train_loader = self._get_loader(mode='train')
        val_loader = self._get_loader(mode='val')
        test_loader = self._get_loader(mode='test')
        return train_loader, val_loader, test_loader

    def _get_loader(self, mode='train'):
        load_dataset = None
        data = self.data[mode]
        if self.args.is_federated:
            loader = {}
            for client in data.keys():
                if mode == 'train':
                    load_dataset = CFTrainDataset({client: data[client]}, {client: self.negatives[client]})
                elif mode == 'val' or mode == 'test':
                    load_dataset = CFEvalDataset({client: data[client]}, {client: self.data['train'][client]})

                loader[client] = data_utils.DataLoader(load_dataset, batch_size=self.args.batch_size,
                                                       shuffle=False, pin_memory=True)
        else:
            if mode == 'train':
                load_dataset = CFTrainDataset(data, self.negatives)
            elif mode == 'val' or mode == 'test':
                load_dataset = CFEvalDataset(data, self.data['train'])

            loader = data_utils.DataLoader(load_dataset, batch_size=self.args.batch_size,
                                           shuffle=False, pin_memory=True)

        return loader
