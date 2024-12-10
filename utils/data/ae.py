import copy

import torch
import torch.utils.data as data_utils
from scipy import sparse
import numpy as np
from .abstractClass import AbstractDataLoader


class AETrainDataset(data_utils.Dataset):

    def __init__(self, user2items, item_count):
        # Row indices for sparse matrix
        #   e.g. [0, 0, 0, 1, 1, 4, 4, 4, 4]
        #        when user2items = {0:[1,2,3], 1:[4,5], 4:[6,7,8,9]}

        user_row = []
        for user, items in enumerate(user2items.values()):
            for _ in range(len(items)):
                user_row.append(user)

        # Column indices for sparse matrix
        item_col = []
        for items in user2items.values():
            item_col.extend(items)

        # Construct sparse matrix
        assert len(user_row) == len(item_col)
        sparse_data = sparse.csr_matrix((np.ones(len(user_row)), (user_row, item_col)),
                                        dtype='float64', shape=(len(user2items), item_count))

        # Convert to torch tensor
        self.data = torch.FloatTensor(sparse_data.toarray())

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


def split_input_label_proportion(user2items, scheme='holdout', labels=None):
    """
    Split each user's items to input and label s.t. the two are disjoint
    """

    input_list, label_list = [], []

    if scheme == 'holdout':
        for items in user2items.values():
            split_point = len(items) // 2
            input_list.append(items[:split_point])
            label_list.append(items[split_point:])
    elif scheme == 'leave_one_out':
        for user, items in user2items.items():
            input_list.append(labels[user])
            label_list.append(items)

    return input_list, label_list


class AEEvalDataset(data_utils.Dataset):

    def __init__(self, user2items, item_count, scheme='holdout', labels=None):
        # Split each user's items to input and label s.t. the two are disjoint
        # Both are lists of np.ndarrays
        input_list, label_list = split_input_label_proportion(user2items, scheme, labels)

        # Row indices for sparse matrix
        input_user_row, label_user_row = [], []
        for user, input_items in enumerate(input_list):
            for _ in range(len(input_items)):
                input_user_row.append(user)
        for user, label_items in enumerate(label_list):
            for _ in range(len(label_items)):
                label_user_row.append(user)
        input_user_row, label_user_row = np.array(input_user_row), np.array(label_user_row)

        # Column indices for sparse matrix
        input_item_col = np.hstack(input_list)
        label_item_col = np.hstack(label_list)

        # Construct sparse matrix
        sparse_input = sparse.csr_matrix((np.ones(len(input_user_row)), (input_user_row, input_item_col)),
                                         dtype='float64', shape=(len(input_list), item_count))
        sparse_label = sparse.csr_matrix((np.ones(len(label_user_row)), (label_user_row, label_item_col)),
                                         dtype='float64', shape=(len(label_list), item_count))

        # Convert to torch tensor
        self.input_data = torch.FloatTensor(sparse_input.toarray())
        self.label_data = torch.FloatTensor(sparse_label.toarray())

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        return self.input_data[index], self.label_data[index]


class AEDataLoader(AbstractDataLoader):
    """
        For autoencoders, we should remove users from val/test sets,
        such that rated items NOT in the training set
    """

    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        # extract a list of unique items from the training set
        unique_items = set()
        for items in self.train.values():
            unique_items.update(items)

        # Then, we remove users from the val/test set.
        self.val = {user: items for user, items in self.val.items() \
                    if all(item in unique_items for item in items)}
        self.test = {user: items for user, items in self.test.items() \
                     if all(item in unique_items for item in items)}

        # re-map items
        self.smap = {s: i for i, s in enumerate(unique_items)}
        remap = lambda items: [self.smap[item] for item in items]

        self.data = {
            'train': {user: remap(items) for user, items in self.train.items()},
            'val': {user: remap(items) for user, items in self.val.items()},
            'test': {user: remap(items) for user, items in self.test.items()}
        }

        # some bookkeeping
        self.stats['num']['items'] = len(unique_items)
        self.item_count = len(unique_items)
        args.num_items = self.item_count

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
                    load_dataset = AETrainDataset({client: data[client]}, item_count=self.item_count)
                elif mode == 'val' or mode == 'test':
                    load_dataset = AEEvalDataset({client: data[client]}, item_count=self.item_count,
                                                 scheme=self.args.split, labels={client: self.data['train'][client]})

                loader[client] = data_utils.DataLoader(load_dataset, batch_size=self.args.batch_size,
                                                       shuffle=False, pin_memory=True)
        else:
            if mode == 'train':
                load_dataset = AETrainDataset(data, item_count=self.item_count)
            elif mode == 'val' or mode == 'test':
                load_dataset = AEEvalDataset(data, item_count=self.item_count, scheme=self.args.split,
                                             labels=self.data['train'])

            loader = data_utils.DataLoader(load_dataset, batch_size=self.args.batch_size,
                                           shuffle=False, pin_memory=True)

        return loader
