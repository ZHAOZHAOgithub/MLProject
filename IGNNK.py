import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """

    def __init__(self, in_channels, out_channels, orders, activation='relu'):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                                     out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()
        self.dropout = nn.Dropout(p=0.5)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size = X.shape[0]  # batch_size
        num_node = X.shape[1]
        input_size = X.size(2)  # time_length
        supports = []
        supports.append(A_q)
        supports.append(A_h)

        x0 = X.permute(1, 2, 0)  # (num_nodes, num_times, batch_size)
        x0 = torch.reshape(x0, shape=[num_node, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)
        for support in supports:
            x1 = torch.mm(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, num_node, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)

        x += self.bias

        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)
        x = self.dropout(x)
        return x



class IGNNK(nn.Module):
    """
    GNN on ST datasets to reconstruct the datasets
   x_s
    |GNN_3
   H_2 + H_1
    |GNN_2
   H_1
    |GNN_1
  x^y_m
    """

    def __init__(self, h, z, k, n):
        super(IGNNK, self).__init__()
        self.time_dimension = h
        self.hidden_dimnesion = z
        self.order = k

        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimnesion, self.order)
        self.GNN2 = D_GCN(self.hidden_dimnesion, self.hidden_dimnesion, self.order)
        self.GNN3 = D_GCN(self.hidden_dimnesion, self.time_dimension, self.order, activation='linear')
        self.nodevec1 =nn.Parameter(torch.randn(n, 2), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(n, 2), requires_grad=True)
        self.nodevec3 = nn.Parameter(torch.randn(n, 2), requires_grad=True)
        self.nodevec4 = nn.Parameter(torch.randn(n, 2), requires_grad=True)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes) # batch_size 时间长 节点数量 12 或 8
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        X = X.to(torch.float32)
        A_q = torch.mm(self.nodevec1, self.nodevec2.permute(1,0))
        A_h = torch.mm(self.nodevec3, self.nodevec4.permute(1,0))
        X_S = X.permute(0, 2, 1)  # to correct the input dims batch 节点数量 时间长

        X_s1 = self.GNN1(X_S, A_q, A_h)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1  # num_nodes, rank
        X_s3 = self.GNN3(X_s2, A_q, A_h)

        #X_res = X_s3.permute(0, 2, 1)
        X_res = X_s3[:,-1]
        return X_res


class IGNNK_label(nn.Module):
    """
    GNN on ST datasets to reconstruct the datasets
   x_s
    |GNN_3
   H_2 + H_1
    |GNN_2
   H_1
    |GNN_1
  x^y_m
    """

    def __init__(self, h, z, k, n):
        super(IGNNK_label, self).__init__()
        self.time_dimension = h
        self.hidden_dimnesion = z
        self.order = k

        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimnesion, self.order)
        self.GNN2 = D_GCN(self.hidden_dimnesion, self.hidden_dimnesion, self.order)
        self.GNN3 = D_GCN(self.hidden_dimnesion, self.time_dimension, self.order, activation='linear')
        self.nodevec1 =nn.Parameter(torch.randn(n, 3), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(n, 3), requires_grad=True)
        self.nodevec3 = nn.Parameter(torch.randn(n, 3), requires_grad=True)
        self.nodevec4 = nn.Parameter(torch.randn(n, 3), requires_grad=True)
        self.sig = nn.Sigmoid()

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes) # batch_size 时间长 节点数量 12 或 8
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        X = X.to(torch.float32)
        A_q = torch.mm(self.nodevec1, self.nodevec2.permute(1,0))
        A_h = torch.mm(self.nodevec3, self.nodevec4.permute(1,0))
        X_S = X.permute(0, 2, 1)  # to correct the input dims batch 节点数量 时间长

        X_s1 = self.GNN1(X_S, A_q, A_h)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1  # num_nodes, rank
        X_s3 = self.GNN3(X_s2, A_q, A_h)
        X_s3 = self.sig(X_s3)
        #X_res = X_s3.permute(0, 2, 1)
        X_res = X_s3[:,-1]
        return X_res