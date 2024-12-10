import copy
import logging
import os
import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp


def dataset_filter(ratings, min_items=5):
    """
            Only keep the data useful, which means:
                - all ratings are non-zeros
                - each user rated at least {self.min_items} items
            :param ratings: pd.DataFrame
            :param min_items: the least number of items user rated
            :return: filter_ratings: pd.DataFrame
            """

    # filter unuseful data
    ratings = ratings[ratings['rating'] > 0]

    # only keep users who rated at least {self.min_items} items
    user_count = ratings.groupby('uid').size()
    user_subset = np.in1d(ratings.uid, user_count[user_count >= min_items].index)
    filter_ratings = ratings[user_subset].reset_index(drop=True)

    del ratings

    return filter_ratings


class ResetDataFrame(object):

    def __init__(self):
        """
        Reset the user ID and item ID in the DataFrame.
        """
        logging.debug("# Initialize a ResetDataFrame object")
        self.item_encoder = LabelEncoder()
        self.user_encoder = LabelEncoder()

    def fit_transform(self, df):
        """
        Reset the user ID and item ID in the DataFrame.
        :param df: pd.DataFrame
        :return: pd.DataFrame
        """
        logging.debug("# Resetting user IDs and item IDs in the DataFrame")

        df['userId'] = self.user_encoder.fit_transform(df['userId'])
        df['itemId'] = self.user_encoder.fit_transform(df['itemId'])

        return df

    def inverse_transform(self, df):
        """
        Inverse the reset operation.
        :param df: pd.DataFrame
        :return: pd.DataFrame
        """
        df['userId'] = self.user_encoder.inverse_transform(df['userId']) - 1
        df['userId'] = self.user_encoder.inverse_transform(df['itemId']) - 1

        return df


class Dataset(object):

    def __init__(self, args):

        self.args = args

        self.ratings = self._load_data()

        self.stats = self._get_statistics()

        args.num_users = self.stats['num']['users']
        args.num_items = self.stats['num']['items']

        # get the index pool of userId and itemId
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())

        # process the ratings data
        self.preprocess_ratings = self._process_ratings()

        logging.debug("# Generate User History Data")
        self.history = self.ratings.groupby('userId').itemId.apply(list).to_dict()

        self.train, self.val, self.test = self._split_dataset()

        logging.debug('# Negative Sampling')
        self.negatives = self._sample_negative()

        if args.alias == 'LightGCN':
            self.adj_mat = self._create_sparse_matrix()

    def _load_data(self):
        """
        Load data from path
        """

        min_rates = 10

        dataset = self.args.dataset
        dataset_file = os.path.join(self.args.data_path, self.args.dataset, self.args.data_file)

        if dataset == "movielens":
            ratings = pd.read_csv(dataset_file, sep=',', header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                                  engine='python')
        elif dataset == "amazon":
            ratings = pd.read_csv(dataset_file, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                                  engine='python')

        elif dataset == "books":

            min_rates = 5

            ratings = pd.read_csv(dataset_file, sep=",", header=1, usecols=[3, 4, 6], names=['uid', 'mid', 'rating'],
                                  engine='python')

            # take the item orders instead of real timestamp
            rank = ratings[['mid']].drop_duplicates().reindex()
            rank['timestamp'] = np.arange((len(rank)))
            ratings = pd.merge(ratings, rank, on=['mid'], how='left')

        elif dataset == "last.fm":
            min_rates = 10

            ratings = pd.read_csv(dataset_file, sep="\t", header=None, usecols=[0, 1, 2],
                                  names=['uid', 'mid', 'rating'],
                                  engine='python')

            # take the item orders instead of real timestamp
            rank = ratings[['mid']].drop_duplicates().reindex()
            rank['timestamp'] = np.arange((len(rank)))
            ratings = pd.merge(ratings, rank, on=['mid'], how='left')


        elif dataset == "user-behavior":
            chunks = pd.read_csv(dataset_file, sep=",", header=None,
                                 names=['uid', 'mid', 'cid', 'behavior', 'timestamp'],
                                 engine='python', chunksize=1000000)

            all_chunks = []
            for chunk in chunks:
                chunk.loc[chunk['behavior'] == 'pv', 'rating'] = 1
                chunk.loc[chunk['behavior'] == 'cart', 'rating'] = 2
                chunk.loc[chunk['behavior'] == 'fav', 'rating'] = 3
                chunk.loc[chunk['behavior'] == 'buy', 'rating'] = 4
                all_chunks.append(chunk)

            ratings = pd.concat(all_chunks)

        elif dataset == "tenrec":

            chunks = pd.read_csv(dataset_file, sep=",", header=1, usecols=[0, 1, 2],
                                 names=['uid', 'mid', 'rating'],
                                 engine='python', chunksize=1000000)

            all_chunks = []
            for chunk in chunks:
                all_chunks.append(chunk)

            ratings = pd.concat(all_chunks)

            # take the item orders instead of real timestamp
            rank = ratings[['mid']].drop_duplicates().reindex()
            rank['timestamp'] = np.arange((len(rank)))
            ratings = pd.merge(ratings, rank, on=['mid'], how='left')
        elif dataset == 'tafeng':
            ratings = pd.read_csv(dataset_file, sep=",", header=1, usecols=[1, 5, 0],
                                  names=['uid', 'mid', 'timestamp'], engine='python')
            ratings['rating'] = 1
        else:
            ratings = pd.DataFrame()

        ratings = dataset_filter(ratings, min_rates)

        # Reindex user id and item id
        user_id = ratings[['uid']].drop_duplicates().reindex()
        user_id['userId'] = np.arange(len(user_id))
        ratings = pd.merge(ratings, user_id, on=['uid'], how='left')

        item_id = ratings[['mid']].drop_duplicates()
        item_id['itemId'] = np.arange(len(item_id))
        ratings = pd.merge(ratings, item_id, on=['mid'], how='left')

        ratings = ratings[['userId', 'itemId', 'rating', 'timestamp']].sort_values(by='userId', ascending=True)

        reset_df = ResetDataFrame()
        ratings = reset_df.fit_transform(ratings)

        return ratings

    def _get_statistics(self):
        """
        Get the statistics of the dataset.
        :return:
        """

        maxs = self.ratings.max()
        seq_len = self.ratings.groupby('userId').size()

        stats = {
            'num': {
                'users': int(maxs['userId']) + 1,
                'items': int(maxs['itemId']) + 1,
                'interactions': len(self.ratings)
            },
            'sequence': {
                'median': int(seq_len.median()),
                'average': seq_len.mean(),
                'min': int(seq_len.min()),
                'max': int(seq_len.max())
            }
        }

        stats['sparsity'] = 1 - stats['num']['interactions'] / (
                stats['num']['users'] * stats['num']['items']
        )

        # print statistic
        logging.info('[{}/{}] Statistics: [#Users: {}, #Items: {}, #Interactions: {}, Sparsity: {:.2f}%]'.format(
            self.args.dataset, self.args.data_file.split('.')[0],
            stats['num']['users'], stats['num']['items'], stats['num']['interactions'], stats['sparsity'] * 100
        ))

        logging.info('[{}/{}] The sequence length: [median: {:d}, average: {:.2f}, min: {:d}, max: {:d}].'.format(
            self.args.dataset, self.args.data_file.split('.')[0],
            stats['sequence']['median'], stats['sequence']['average'],
            stats['sequence']['min'], stats['sequence']['max']
        ))

        return stats

    def _process_ratings(self):
        """
        Process the ratings data according to the data type.
        """

        data = copy.deepcopy(self.ratings)

        if self.args.data_type == 'implicit':
            # binarize into 0 or 1 for implicit feedback
            data.loc[data['rating'] > 0, 'rating'] = 1.0

        elif self.args.data_type == 'explicit':
            # normalize into [0, 1] for explicit feedback
            max_rating = data.ratings.max()
            data['rating'] = data.ratings * 1.0 / max_rating

        return data

    def _sample_negative(self):
        interact_status = self.preprocess_ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def _split_dataset(self):

        ratings = copy.deepcopy(self.preprocess_ratings)

        logging.debug('Splitting dataset following the scheme: {}...'.format(self.args.split))

        if self.args.split == 'leave_one_out':
            user_group = ratings.groupby('userId')
            # user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['itemId']))
            interact_items = user_group.apply(lambda d: list(d.sort_values(by='timestamp')['itemId']))
            train, val, test = {}, {}, {}
            for user in range(self.stats['num']['users']):
                items = interact_items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
        elif self.args.split == 'holdout':

            holdout_size = max(int(self.args.holdout_rates * self.stats['num']['users']), 100)

            # Generate user indices
            permuted_idx = np.random.permutation(self.stats['num']['users'])
            train_idx = permuted_idx[:-2 * holdout_size]
            val_idx = permuted_idx[-2 * holdout_size:-holdout_size]
            test_idx = permuted_idx[-holdout_size:]

            # Split DataFrames
            train_df = ratings.loc[ratings['userId'].isin(train_idx)]
            val_df = ratings.loc[ratings['userId'].isin(val_idx)]
            test_df = ratings.loc[ratings['userId'].isin(test_idx)]

            # DataFrame to dict => {uid : list of sid's}
            train = dict(train_df.groupby('userId').apply(lambda d: list(d['itemId'])))
            val = dict(val_df.groupby('userId').apply(lambda d: list(d['itemId'])))
            test = dict(test_df.groupby('userId').apply(lambda d: list(d['itemId'])))

        else:
            raise ValueError('Invalid split type: {}'.format(self.args.split))

        return train, val, test

    def get_split_dataset(self):
        return self.train, self.val, self.test

    def _create_sparse_matrix(self):

        num_nodes = self.stats['num']['users'] + self.stats['num']['items']

        user_np = np.array(self.ratings['userId'].values, dtype=np.int32)
        item_np = np.array(self.ratings['itemId'].values, dtype=np.int32)
        ratings = np.array(self.ratings['rating'].values, dtype=np.int32)

        tmp_adj = sp.csr_matrix(
            (ratings, (user_np, item_np + self.stats['num']['users'])),
            shape=(num_nodes, num_nodes)
        )

        adj_mat = tmp_adj + tmp_adj.T

        # normalize matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        # normalize by user counts
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        # normalize by item counts
        normalized_adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # convert to torch sparse matrix
        adj_mat_coo = normalized_adj_matrix.tocoo()

        values = adj_mat_coo.data
        indices = np.vstack((adj_mat_coo.row, adj_mat_coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj_mat_coo.shape

        return torch.sparse_coo_tensor(i, v, torch.Size(shape))
