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


def load_data(root="/mnt/data/wangzhu/pygcn", path="/data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    #print(f'{path}, {dataset}')

    idx_features_labels = np.genfromtxt(root + "{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    #index = [i for i in range(len(idx))]
    #np.random.shuffle(index)
    idx_map = {j: i for i, j in enumerate(idx)}     
    edges_unordered = np.genfromtxt(root + "{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix 对称转置矩阵（无向图）
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features) #对特征（初始）矩阵归一化
    adj = normalize(adj + sp.eye(adj.shape[0])) #对A+I归一化

    #dataset split 划分训练，验证，测试数据
    #按照指定数量划分
    '''idx_train = range(140)
    idx_val = range(200,500)
    idx_test = range(500,1500)
    '''
    #按照比例进行数据集划分0.6/0.2/0.2
    train_ratio = 0.6
    val_ratio = 0.2  
    #test_ratio = 1-train_ratio-val_ratio
    train_coun = round(features.shape[0] * train_ratio)
    val_end = train_coun + round(features.shape[0] * val_ratio)
    test_end = features.shape[0]
    idx_train = range(train_coun)
    idx_val = range(train_coun, val_end)
    idx_test = range(val_end, test_end)
    print('train_coun:',train_coun, 
            'validation end:', val_end, 'test end:', test_end)
    
    #numpy数据格式转换为torch（tensor）
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj) #邻接矩阵转换tensor，自定义函数for稀疏矩阵

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0. #将倒数nan归为0
    r_mat_inv = sp.diags(r_inv) #将n*1矩阵改为 n*n矩阵
    mx = r_mat_inv.dot(mx) #计算归一化矩阵，对A来说简化过程D-1*A
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
