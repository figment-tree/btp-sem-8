#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from .model_base import Info, Model


def graph_generating(raw_graph, row, col):
    if raw_graph.shape == (row, col):
        graph = sp.bmat([[sp.identity(raw_graph.shape[0]), raw_graph],
                         [raw_graph.T, sp.identity(raw_graph.shape[1])]])
    else:
        raise ValueError(r"raw_graph's shape is wrong")
    return graph


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices),
                                     torch.FloatTensor(values),
                                     torch.Size(graph.shape))
    return graph


class GCN_Info(Info):
    def __init__(self, embedding_size, embed_L2_norm, mess_dropout,
                 node_dropout, num_layers, act=nn.LeakyReLU()):
        super().__init__(embedding_size, embed_L2_norm)
        self.act = act
        assert 1 > mess_dropout >= 0
        self.mess_dropout = mess_dropout
        assert 1 > node_dropout >= 0
        self.node_dropout = node_dropout
        assert isinstance(num_layers, int) and num_layers > 0
        self.num_layers = num_layers


class GCN(Model):
    def get_infotype(self):
        return GCN_Info

    def __init__(self, info, dataset, raw_graph, device, pretrain=None):
        super().__init__(info, dataset, create_embeddings=True)
        self.items_feature = nn.Parameter(
            torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)

        self.epison = 1e-8

        assert isinstance(raw_graph, list)
        ul_graph, li_graph = raw_graph

        li_norm = sp.diags(
            1 / (np.sqrt((li_graph.multiply(li_graph)).sum(axis=1).A.ravel()) + 1e-8)) @ li_graph
        #  List-item-list metapath
        ll_graph = li_norm @ li_norm.T

        if ul_graph.shape == (self.num_users, self.num_lists):
            # add self-loops
            atom_graph = sp.bmat([[sp.identity(ul_graph.shape[0]), ul_graph],
                                  [ul_graph.T, sp.identity(ul_graph.shape[1])]])
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        self.atom_graph = to_tensor(laplace_transform(atom_graph)).to(device)
        print('finish generating atom graph')

        if li_graph.shape == (self.num_lists, self.num_items) \
                and ll_graph.shape == (self.num_lists, self.num_lists):
            # add self-loop
            non_atom_graph = sp.bmat([[ll_graph, li_graph],
                                      [li_graph.T, sp.identity(li_graph.shape[1])]])
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        self.non_atom_graph = to_tensor(
            laplace_transform(non_atom_graph)).to(device)
        print('finish generating non-atom graph')

        # copy from info
        self.act = self.info.act
        self.num_layers = self.info.num_layers
        self.device = device

        #  Dropouts
        self.mess_dropout = nn.Dropout(self.info.mess_dropout, True)
        self.node_dropout = nn.Dropout(self.info.node_dropout, True)

        # Layers
        self.dnns_atom = nn.ModuleList([nn.Linear(
            self.embedding_size * (l + 1), self.embedding_size) for l in range(self.num_layers)])
        self.dnns_non_atom = nn.ModuleList([nn.Linear(
            self.embedding_size * (l + 1), self.embedding_size) for l in range(self.num_layers)])

        # pretrain
        if pretrain is not None:
            self.users_feature.data = F.normalize(
                pretrain['users_feature'])
            self.items_feature.data = F.normalize(
                pretrain['items_feature'])
            self.bundles_feature.data = F.normalize(
                pretrain['bundles_feature'])

    def one_propagate(self, graph, A_feature, B_feature, attns, dnns):
        # node dropout on graph
        indices = graph._indices()
        values = graph._values()
        values = self.node_dropout(values)
        graph = torch.sparse.FloatTensor(indices, values, size=graph.shape)

        # propagate
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]
        for i in range(self.num_layers):
            features = self.mess_dropout(torch.cat([self.act(
                dnns[i](torch.matmul(graph, features))), features], 1))
            all_features.append(F.normalize(features))

        all_features = torch.cat(all_features, 1)
        A_feature, B_feature = torch.split(
            all_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        return A_feature, B_feature

    def propagate(self):
        #  ======================  user-list propagation  ======================
        atom_users_feature, atom_bundles_feature = self.one_propagate(
            self.atom_graph, self.users_feature, self.bundles_feature,
            None, self.dnns_atom)

        #  ======================= list-item propagation =======================
        non_atom_bundles_feature, _ = self.one_propagate(
            self.non_atom_graph, self.bundles_feature, self.items_feature,
            None, self.dnns_non_atom)

        users_feature = atom_users_feature
        bundles_feature = [atom_bundles_feature, non_atom_bundles_feature]

        return users_feature, bundles_feature

    def predict(self, users_feature, bundles_feature):
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature  # batch_n_f
        pred = torch.sum(
            users_feature * (bundles_feature_atom + bundles_feature_non_atom),
            2)
        return pred

    def forward(self, users, lists):
        users_feature, bundles_feature = self.propagate()
        # u_f --> batch_f --> batch_n_f
        users_embedding = users_feature[users]
        lists_embedding = [i[lists]
                           for i in bundles_feature]  # b_f --> batch_n_f
        pred = self.predict(users_embedding, lists_embedding)
        loss = self.regularize(users_embedding, lists_embedding)
        return pred, loss

    def regularize(self, users_feature, bundles_feature):
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature  # batch_n_f
        loss = self.embed_L2_norm * \
            ((users_feature ** 2).sum() + (bundles_feature_atom ** 2).sum() +
             (bundles_feature_non_atom ** 2).sum())
        return loss

    def evaluate(self, propagate_result, users):
        '''
        just for testing, compute scores of all lists for `users` by `propagate_result`
        '''
        users_feature, bundles_feature = propagate_result
        users_feature_atom = users_feature[users]  # batch_f
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature  # b_f
        scores = torch.mm(users_feature_atom,
                          (bundles_feature_atom.t() +
                           bundles_feature_non_atom.t()))  # batch_b
        return scores


class GAT_Info(Info):
    def __init__(self, embedding_size, embed_L2_norm, mess_dropout,
                 node_dropout, num_layers, act=nn.LeakyReLU()):
        super().__init__(embedding_size, embed_L2_norm)
        self.act = act
        assert 1 > mess_dropout >= 0
        self.mess_dropout = mess_dropout
        assert 1 > node_dropout >= 0
        self.node_dropout = node_dropout
        assert isinstance(num_layers, int) and num_layers > 0
        self.num_layers = num_layers


class GAT(Model):
    def get_infotype(self):
        return GAT_Info

    def __init__(self, info, dataset, raw_graph, device, pretrain=None):
        super().__init__(info, dataset, create_embeddings=True)
        self.items_feature = nn.Parameter(
            torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)

        self.epison = 1e-8

        assert isinstance(raw_graph, list)
        ul_graph, li_graph = raw_graph

        li_norm = sp.diags(
            1 / (np.sqrt((li_graph.multiply(li_graph)).sum(axis=1).A.ravel()) + 1e-8)) @ li_graph
        #  List-item-list metapath
        ll_graph = li_norm @ li_norm.T

        if ul_graph.shape == (self.num_users, self.num_lists):
            # add self-loops
            atom_graph = sp.bmat([[sp.identity(ul_graph.shape[0]), ul_graph],
                                  [ul_graph.T, sp.identity(ul_graph.shape[1])]])
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        self.atom_graph = to_tensor(laplace_transform(atom_graph)).to(device)
        print('finish generating atom graph')

        if li_graph.shape == (self.num_lists, self.num_items) \
                and ll_graph.shape == (self.num_lists, self.num_lists):
            # add self-loop
            non_atom_graph = sp.bmat([[ll_graph, li_graph],
                                      [li_graph.T, sp.identity(li_graph.shape[1])]])
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        self.non_atom_graph = to_tensor(
            laplace_transform(non_atom_graph)).to(device)
        print('finish generating non-atom graph')

        # copy from info
        self.act = self.info.act
        self.num_layers = self.info.num_layers
        self.device = device

        #  Dropouts
        self.mess_dropout = nn.Dropout(self.info.mess_dropout, True)
        self.node_dropout = nn.Dropout(self.info.node_dropout, True)

        # Layers
        self.attns_atom = nn.ModuleList([nn.Linear(
            2 * self.embedding_size * (l + 1), self.embedding_size * (l + 1)) for l in range(self.num_layers)])
        self.dnns_atom = nn.ModuleList([nn.Linear(
            self.embedding_size * (l + 1), self.embedding_size) for l in range(self.num_layers)])
        self.attns_non_atom = nn.ModuleList([nn.Linear(
            2 * self.embedding_size * (l + 1), self.embedding_size * (l + 1)) for l in range(self.num_layers)])
        self.dnns_non_atom = nn.ModuleList([nn.Linear(
            self.embedding_size * (l + 1), self.embedding_size) for l in range(self.num_layers)])

        # pretrain
        if pretrain is not None:
            self.users_feature.data = F.normalize(
                pretrain['users_feature'])
            self.items_feature.data = F.normalize(
                pretrain['items_feature'])
            self.bundles_feature.data = F.normalize(
                pretrain['bundles_feature'])

    def one_propagate(self, graph, A_feature, B_feature, attns, dnns):
        # node dropout on graph
        indices = graph._indices()
        values = graph._values()
        values = self.node_dropout(values)
        graph = torch.sparse.FloatTensor(indices, values, size=graph.shape)

        # propagate
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]
        for i in range(self.num_layers):
            attn_weights = F.softmax(
                attns[i](torch.cat((features, features), 1)), dim=0)
            features = attn_weights * features

            features = self.mess_dropout(torch.cat([self.act(
                dnns[i](torch.matmul(graph, features))), features], 1))
            all_features.append(F.normalize(features))

        all_features = torch.cat(all_features, 1)
        A_feature, B_feature = torch.split(
            all_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        return A_feature, B_feature

    def propagate(self):
        #  ======================  user-list propagation  ======================
        atom_users_feature, atom_bundles_feature = self.one_propagate(
            self.atom_graph, self.users_feature, self.bundles_feature,
            self.attns_atom, self.dnns_atom)

        #  ======================= list-item propagation =======================
        non_atom_bundles_feature, _ = self.one_propagate(
            self.non_atom_graph, self.bundles_feature, self.items_feature,
            self.attns_non_atom, self.dnns_non_atom)

        users_feature = atom_users_feature
        bundles_feature = [atom_bundles_feature, non_atom_bundles_feature]

        return users_feature, bundles_feature

    def predict(self, users_feature, bundles_feature):
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature  # batch_n_f
        pred = torch.sum(
            users_feature * (bundles_feature_atom + bundles_feature_non_atom),
            2)
        return pred

    def forward(self, users, lists):
        users_feature, bundles_feature = self.propagate()
        # u_f --> batch_f --> batch_n_f
        users_embedding = users_feature[users]
        lists_embedding = [i[lists]
                           for i in bundles_feature]  # b_f --> batch_n_f
        pred = self.predict(users_embedding, lists_embedding)
        loss = self.regularize(users_embedding, lists_embedding)
        return pred, loss

    def regularize(self, users_feature, bundles_feature):
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature  # batch_n_f
        loss = self.embed_L2_norm * \
            ((users_feature ** 2).sum() + (bundles_feature_atom ** 2).sum() +
             (bundles_feature_non_atom ** 2).sum())
        return loss

    def evaluate(self, propagate_result, users):
        '''
        just for testing, compute scores of all lists for `users` by `propagate_result`
        '''
        users_feature, bundles_feature = propagate_result
        users_feature_atom = users_feature[users]  # batch_f
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature  # b_f
        scores = torch.mm(users_feature_atom,
                          (bundles_feature_atom.t() +
                           bundles_feature_non_atom.t()))  # batch_b
        return scores
