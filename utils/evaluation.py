import copy

import numpy as np
import torch
import math

import pandas as pd


def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks=None):
    if ks is None:
        ks = [5, 10, 20, 50, 100]

    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics


class Metric(object):

    def __init__(self, args):
        self.args = args
        self.topK = args.top_k
        self._subjects = None

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
        # the golden set
        test = pd.DataFrame({'user': test_users,
                             'test_item': test_items,
                             'test_score': test_scores})
        # the full set
        all_data = pd.DataFrame({'user': neg_users + test_users,
                                 'item': neg_items + test_items,
                                 'score': neg_scores + test_scores})
        all_data = pd.merge(all_data, test, on=['user'], how='left')
        # rank the items according to the scores for each user
        all_data['rank'] = all_data.groupby('user')['score'].rank(method='first', ascending=False)
        all_data.sort_values(['user', 'rank'], inplace=True)

        top_k = all_data[all_data['rank'] <= self.topK]
        self.test_in_top_k = top_k[top_k['test_item'] == top_k['item']]

    def call_hit_ratio(self):
        """Hit Ratio @ top_K"""

        return len(self.test_in_top_k) * 1.0 / self.user_num

    def call_ndcg(self):
        """NDCG @ top_K"""

        ndcg = self.test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
        self.test_in_top_k['ndcg'] = ndcg

        return self.test_in_top_k['ndcg'].sum() * 1.0 / self.user_num

    def evaluate(self):
        """
        Evaluate the performance of models
        """

        hr, ndcg = self.call_hit_ratio(), self.call_ndcg()

        return hr, ndcg
