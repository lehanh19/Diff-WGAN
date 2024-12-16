import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
import time
import dgl
from torch.utils.data import Dataset

def data_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), \
        (train_list[:, 0], train_list[:, 1])), dtype='float64', \
        shape=(n_user, n_item))
    
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    # ui_graph = makeBiAdj(train_data, n_user, n_item)
    # ui_graph = getSparseGraph(n_user, n_item, train_data)
    ui_graph = None
    
    return train_data, valid_y_data, test_y_data, n_user, n_item, ui_graph


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        # return index, item
        return item
    def __len__(self):
        return len(self.data)
    
def makeBiAdj(mat, n_user, n_item):
    a = sp.csr_matrix((n_user, n_user))
    b = sp.csr_matrix((n_item, n_item))
    mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
    mat = (mat != 0) * 1.0
    mat = mat.tocoo()
    edge_src,edge_dst = mat.nonzero()
    ui_graph = dgl.graph(data=(edge_src, edge_dst),
                        idtype=torch.int32,
                            num_nodes=mat.shape[0]
                            )
    return ui_graph

def select_negative_items(realData, num_pm=50, num_zr=50):
    '''
    realData : n-dimensional indicator vector specifying whether u has purchased each item i
    num_pm : num of negative items (partial-masking) sampled on the t-th iteration
    num_zr : num of negative items (zero-reconstruction regularization) sampled on the t-th iteration
    '''
    data = copy.deepcopy(realData).cpu().numpy()
    n_items_pm = np.zeros_like(data)
    n_items_zr = np.zeros_like(data)

    for i in range(data.shape[0]):
        p_items = np.where(data[i] != 0)[0]
        # all_item_index = random.sample(range(data.shape[1]), 1683)
        all_item_index = np.arange(data.shape[1]).tolist()
        all_item_index = list(set(all_item_index).difference(list(p_items)))
        # for j in range(p_items.shape[0]):
        #     # if list(p_items)[j] in all_item_index:
        #     all_item_index.remove(list(p_items)[j])
        random.shuffle(all_item_index)
        n_item_index_pm = all_item_index[0 : num_pm]
        n_item_index_zr = all_item_index[num_pm : (num_pm+num_zr)]
        n_items_pm[i][n_item_index_pm] = 1
        n_items_zr[i][n_item_index_zr] = 1
    
    return n_items_pm, n_items_zr

def getSparseGraph(n_users, n_items, UserItemNet):
    print("loading adjacency matrix")
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = UserItemNet.tolil()
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()
    
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    
    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()

    Graph = convert_sp_mat_to_sp_tensor(norm_adj)
    Graph = Graph.coalesce()
    return Graph

def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))