import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str), comments=None)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32) 
    labels = encode_onehot(idx_features_labels[:, -1])
    n_objects = idx_features_labels.shape[0]

    # build graph
    idx = np.array(idx_features_labels[:, 0], str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=str, comments=None)
    edges = np.zeros_like(edges_unordered, dtype=np.int32)
    counter = 0
    for edge in edges_unordered:
        try:
            edges[counter][0] = idx_map[edge[0]]
            edges[counter][1] = idx_map[edge[1]]
            counter += 1
        except:
            pass
    
    edges = edges[0 : counter]
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_2 = adj.multiply(adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj_2 = normalize(adj_2)
    
    if n_objects > 1500:
        idx_train = range(140)
        idx_val = range(140, 500)
        idx_test = range(500, 1500)
    else:
        idx_train = range(int(n_objects * 0.1))
        idx_val = range(int(n_objects * 0.1), int(n_objects * 0.3))
        idx_test = range(int(n_objects * 0.3), n_objects)
        print('n_objects ({}) < 1500'.format(n_objects))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_2 = sparse_mx_to_torch_sparse_tensor(adj_2)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print(features.shape)
    print(n_objects)
    print(edges_unordered.shape)
    return adj, adj_2, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
