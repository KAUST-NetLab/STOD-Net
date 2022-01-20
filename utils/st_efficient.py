# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: st_efficient.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-08-15 (YYYY-MM-DD)
-----------------------------------------------
"""
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from st_dense_gcn.utils.effnet import effnetv2_s

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        # print(self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_out, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return x


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # ct's implementation of tensor multiplication
        self.a_l = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a_r = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        a_l = torch.matmul(h, self.a_l)  # a_l is with shape [batch, N, out_features, 1]
        a_r = torch.matmul(h, self.a_r)  # a_r is with shape [batch, N, out_features, 1]
        # Use the broadcast scheme of pytorch
        # https://discuss.pytorch.org/t/how-to-multiply-each-element-of-a-vector-with-every-element-of-another-vector-using-pytorch/20558
        e = self.leakyrelu(a_l.permute(0, 2, 1) + a_r)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
            # return torch.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_out, dropout, alpha=0.1, nheads=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(n_hid * nheads, n_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.out_att(x, adj)


class DenseNetUnit(nn.Module):
    def __init__(self, growth_rate=12, block_config=(8, 8, 8), compression=0.5, num_init_features=24,
                 bn_size=4, drop_rate=0.2, efficient=True, nb_flow=2,
                 channels=3, steps=1, height=16, width=8,
                 gcn_type='gcn', fusion=False, gcn_layer=3, flow_feature=True, gate_type=1, alpha=0.5, beta=0.5):
        super(DenseNetUnit, self).__init__()
        self.channels = channels
        self.nb_flow = nb_flow
        self.steps = steps
        self.height = height
        self.width = width
        self.gcn_type = gcn_type
        self.gcn_layer = gcn_layer
        self.num_init_features = num_init_features
        self.fusion = fusion
        self.flow_feature = flow_feature
        self.gate_type = gate_type
        self.alpha = alpha
        self.beta = beta

        gnn = GCN if self.gcn_type.lower() == 'gcn' else GAT
        self.gcn_dynamic = dict()
        self.gcn_static = dict()
        self.conv = dict()
        for i in range(gcn_layer):
            self.gcn_dynamic[i] = gnn(self.nb_flow * self.channels if i == 0 else self.num_init_features,
                                      self.num_init_features, self.num_init_features, dropout=0.5).to(device)
            self.gcn_static[i] = gnn(48 * 2 if i == 0 else self.num_init_features,
                                     self.num_init_features, self.num_init_features, dropout=0.5).to(device)
            self.conv[i] = nn.Conv2d(self.nb_flow * self.channels if i == 0 else self.num_init_features,
                                     self.num_init_features, kernel_size=3, padding=1).to(device)

    def forward(self, x, adj=None, feature=None, road_adj=None, road_feature=None):
        x = x.view(-1, self.channels * self.nb_flow, self.height, self.width)
        for i in range(self.gcn_layer):
            # convolutional learning
            x = self.conv[i](x)
            # dynamic OD gating
            # d_out = self.gcn_dynamic[i](feature, adj) if i == 0 else self.gcn_dynamic[i](SiLU()(d_out), adj)
            d_out = self.gcn_dynamic[i](feature, adj) if i == 0 else self.gcn_dynamic[i](d_out, adj)
            d_gate = F.hardsigmoid(d_out)
            # d_gate = self.dynamic_gating[i](d_out)
            d_gate = d_gate.view(-1, self.height, self.width, self.num_init_features).permute(0, 3, 1, 2)
            d_gate = d_gate.contiguous()

            # static OD gating
            s_out = self.gcn_static[i](road_feature, road_adj) if i == 0 else self.gcn_static[i](s_out, road_adj)
            s_gate = F.hardsigmoid(s_out)
            # s_gate = self.static_gating[i](s_out)
            s_gate = s_gate.view(-1, self.height, self.width, self.num_init_features).permute(0, 3, 1, 2)
            s_gate = s_gate.contiguous()

            if self.gate_type == 1:
                x *= d_gate
            elif self.gate_type == 2:
                x *= s_gate
            else:
                x *= (self.alpha * d_gate + self.beta * s_gate)

            # SiLU activation
            x = x * torch.sigmoid(x)

            if self.gate_type == 1:
                flow_input = d_gate
            elif self.gate_type == 2:
                flow_input = s_gate
            else:
                flow_input = self.alpha * d_gate + self.beta * s_gate

        return x, flow_input


class Generator(nn.Module):

    def __init__(self, block_config=[8, 8, 8], nb_flow=2, channels=[3, 3, 3],
                 steps=1, height=16, width=8, gcn_type='gcn',
                 fusion=False, norm='01_sigmoid', gcn_layer=3, flow_feature=True, gate_type=1, alpha=0.5, beta=0.5):

        super(Generator, self).__init__()
        self.close = channels[0]
        self.period = channels[1]
        self.trend = channels[2]
        self.nb_flow = nb_flow
        self.gcn_type = gcn_type
        self.norm = norm
        self.fusion = fusion

        self.close_net = DenseNetUnit(block_config=block_config, channels=self.close,
                                      steps=steps, height=height, width=width,
                                      nb_flow=nb_flow,
                                      gcn_type=self.gcn_type, fusion=fusion,
                                      gcn_layer=gcn_layer, flow_feature=flow_feature,
                                      gate_type=gate_type, alpha=alpha, beta=beta) if self.close > 0 else None
        self.period_net = DenseNetUnit(block_config=block_config, channels=self.period,
                                       steps=steps, height=height, width=width,
                                       nb_flow=nb_flow,
                                       gcn_type=self.gcn_type, fusion=fusion,
                                       gcn_layer=gcn_layer, flow_feature=flow_feature,
                                       gate_type=gate_type, alpha=alpha, beta=beta) if self.period > 0 else None
        self.trend_net = DenseNetUnit(block_config=block_config, channels=self.trend,
                                      steps=steps, height=height, width=width,
                                      nb_flow=nb_flow,
                                      gcn_type=self.gcn_type, fusion=fusion,
                                      gcn_layer=gcn_layer, flow_feature=flow_feature,
                                      gate_type=gate_type, alpha=alpha, beta=beta) if self.trend > 0 else None
        self.features = effnetv2_s().to(device)

    def forward(self, vx, adj=None, feature=None, road_net=None, road_feature=None):
        output = 0
        volume_out = []
        flow_out = []
        flow = 0
        if self.close > 0:
            close_out, close_flow = self.close_net(vx[:, :self.close], adj[:, 0],
                                                   feature[:, 0, :, :self.close * self.nb_flow],
                                                   road_net, road_feature)
            output += close_out
            flow += close_flow
        if self.period > 0:
            period_out, period_flow = self.period_net(vx[:, self.close:self.close + self.period],
                                                      adj[:, 1],
                                                      feature[:, 1, :,
                                                      self.close*self.nb_flow:(self.close+self.period)*self.nb_flow],
                                                      road_net, road_feature)
            output += period_out
            flow += period_flow
        if self.trend > 0:
            trend_out, trend_flow = self.trend_net(vx[:, self.close + self.period:], adj[:, 2],
                                                   feature[:, 2, :, (self.close + self.period) * self.nb_flow:],
                                                   road_net, road_feature)
            output += trend_out
            flow += trend_flow

        # out = torch.cat(volume_out, dim=1)
        # flow = torch.cat(flow_out, dim=1)

        output = self.features([output, flow])

        if self.norm.lower() == '01_sigmoid':
            return torch.sigmoid(output)
        elif self.norm.lower() == '11_tanh':
            return torch.tanh(output)
        elif self.norm.lower() == 'z_score_linear':
            return output
        elif self.norm.lower() == '01_tanh':
            return torch.tanh(output)
        else:
            raise Exception('Wrong Norm')
