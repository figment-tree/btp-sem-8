#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset


def sparse_ones(indices, size, dtype=torch.float):
    one = torch.ones(indices.shape[1], dtype=dtype)
    return torch.sparse.FloatTensor(indices, one, size=size).to(dtype)


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices),
                                     torch.FloatTensor(values),
                                     torch.Size(graph.shape))
    return graph


def print_statistics(X, string):
    print('>' * 10 + string + '>' * 10)
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice) / X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice) / X.shape[1])
    print('Matrix density', len(nonzero_row_indice) /
          (X.shape[0] * X.shape[1]))


class BasicDataset(Dataset):
    '''
    generate dataset from raw *.txt
    contains:
        tensors like (`user`, `list_p`, `list_n1`, `list_n2`, ...)
        for BPR
    Args:
    - `path`: the path of dir that contains dataset dir
    - `name`: the name of dataset (used as the name of dir)
    - `num_negative_samples`: the number of negative samples for each user-list_p pair
    - `seed`: seed of `np.random`
    '''

    def __init__(self, path, name, task, num_negative_samples):
        self.path = path
        self.name = name
        self.task = task
        self.num_negative_samples = num_negative_samples
        (self.num_items,
         self.num_lists, self.num_users) = self.__load_data_size()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __load_data_size(self):
        with open(os.path.join(self.path, self.name,
                               '{}_data_size.txt'.format(self.name)),
                  'r') as f:
            return [int(n) for n in f.readline().split('\t')][:3]

    def load_U_L_interaction(self):
        with open(os.path.join(self.path, self.name,
                               f'{self.task}.txt'), 'r') as f:
            user_nodes, list_nodes, edge_weights = [], [], []
            for line in f.readlines():
                user, item_list, edge_weight = line.strip().split("\t")
                user_nodes.append(int(user))
                list_nodes.append(int(item_list))
                edge_weights.append(float(edge_weight))
            return (user_nodes, list_nodes, edge_weights)

    def load_L_I_affiliation(self):
        with open(os.path.join(self.path, self.name,
                               'listItem_{}.txt'.format(self.name)), 'r') as f:
            list_nodes, item_nodes = [], []
            for line in f.readlines():
                item_list, items = line.split("\t")
                item_list = int(item_list)
                items = eval(items)
                for item_node in items:
                    list_nodes.append(item_list)
                    item_nodes.append(item_node)
            return (list_nodes, item_nodes)


class ListTrainDataset(BasicDataset):
    def __init__(self, path, name, seed=None):
        super().__init__(path, name, 'train', 1)
        # User-List
        self.users, self.lists, edge_weights = self.load_U_L_interaction()
        self.__length = len(edge_weights)
        self.ground_truth_u_l = sp.coo_matrix(
            (edge_weights, (self.users, self.lists)),
            shape=(self.num_users, self.num_lists)).tocsr()

        print_statistics(self.ground_truth_u_l, 'U-L statistics in train')

    def __getitem__(self, index):
        user, positive_list = self.users[index], self.lists[index]
        all_lists = [positive_list]
        while True:
            i = np.random.randint(self.num_lists)
            if self.ground_truth_u_l[user, i] == 0 and i not in all_lists:
                all_lists.append(i)
                if len(all_lists) == self.num_negative_samples + 1:
                    break

        return torch.LongTensor([user]), torch.LongTensor(all_lists)

    def __len__(self):
        return self.__length


class ListTestDataset(BasicDataset):
    def __init__(self, path, name, train_dataset, task='test'):
        super().__init__(path, name, task, None)
        # User-List
        self.users, self.lists, edge_weights = self.load_U_L_interaction()
        self.__length = len(edge_weights)
        self.ground_truth_u_l = sp.coo_matrix(
            (edge_weights, (self.users, self.lists)),
            shape=(self.num_users, self.num_lists)).tocsr()

        print_statistics(self.ground_truth_u_l, 'U-L statistics in test')

        self.train_mask_u_l = train_dataset.ground_truth_u_l
        self.users = torch.arange(
            self.num_users, dtype=torch.long).unsqueeze(dim=1)
        self.lists = torch.arange(self.num_lists, dtype=torch.long)
        assert self.train_mask_u_l.shape == self.ground_truth_u_l.shape

    def __getitem__(self, index):
        return index, torch.from_numpy(self.ground_truth_u_l[index].toarray()).squeeze(),  \
            torch.from_numpy(self.train_mask_u_l[index].toarray()).squeeze(),  \


    def __len__(self):
        return self.ground_truth_u_l.shape[0]


class AssistDataset(BasicDataset):
    def __init__(self, path, name):
        super().__init__(path, name, None, None)
        # List-Item
        lists, items = self.load_L_I_affiliation()
        edge_weights = np.ones(len(lists), dtype=np.float32)
        self.ground_truth_l_i = sp.coo_matrix(
            (edge_weights, (lists, items)),
            shape=(self.num_lists, self.num_items)).tocsr()

        print_statistics(self.ground_truth_l_i, 'L-I statistics')


def get_dataset(path, name, task='validation', seed=123):
    assist_data = AssistDataset(path, name)
    print('finish loading list-item affiliation data')

    list_train_data = ListTrainDataset(
        path, name, seed=seed)
    print('finish loading item_list train data')
    list_test_data = ListTestDataset(
        path, name, list_train_data, task=task)
    print('finish loading item_list test data')

    # return list_train_data, list_test_data, item_data, assist_data
    return list_train_data, list_test_data, assist_data
