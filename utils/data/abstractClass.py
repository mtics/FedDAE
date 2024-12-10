import copy
from abc import *


class AbstractDataLoader(metaclass=ABCMeta):

    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.train, self.val, self.test = dataset.get_split_dataset()
        self.stats = dataset.stats

        self.user_count = self.stats['num']['users']
        self.item_count = self.stats['num']['items']

    @abstractmethod
    def get_dataloaders(self):
        pass






