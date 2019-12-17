import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, second_order=False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, second_order=second_order)
        self.gc2 = GraphConvolution(nhid, nclass, second_order=second_order)
        self.dropout = dropout

    def forward(self, x, adj, adj_2=None):
        x = F.relu(self.gc1(x, adj, adj_2))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, adj_2)
        return F.log_softmax(x, dim=1)
