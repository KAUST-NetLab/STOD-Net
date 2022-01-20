# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: st_models_v2.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-08-28 (YYYY-MM-DD)
-----------------------------------------------
"""
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from st_dense_gcn.utils.effv1 import EfficientNet
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
            return torch.relu(h_prime)
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


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concatenated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concatenated_features)))
        return bottleneck_output

    return bn_function


def conv_3x3_bn(inp, oup, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel, padding=padding, bias=False),
        nn.BatchNorm2d(oup)
    )


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, padding=0, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.channels = growth_rate + num_input_features
        self.fc1 = nn.Linear(self.channels, self.channels // 16)
        self.fc2 = nn.Linear(self.channels // 16, self.channels)

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        # add SE Operation
        prev = torch.cat(prev_features, 1)
        new_features = torch.cat((prev, new_features), 1)
        se = self.global_avg(new_features)
        se = se.view(se.size(0), -1)
        se = torch.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        se = se.view(se.size(0), se.size(1), 1, 1)

        new_features = new_features * se

        return new_features


class _TransitionTP(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionTP, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, padding=0,
                                          dilation=1, bias=False))

    def forward(self, x):
        traffic_feature, flow_features = x
        return self.conv(self.relu(self.norm(traffic_feature))), flow_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False,
                 num_init=12, flow_feature=False, gcn_layer=1):
        super(_DenseBlock, self).__init__()
        self.flow_feature = flow_feature
        self.gcn_layer = gcn_layer
        if self.flow_feature & (self.gcn_layer > 0):
            self.num_init = num_init
        else:
            self.num_init = 0

        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate + self.num_init,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        init_features, flow_features = x
        if self.flow_feature & (self.gcn_layer > 0):
            vf_features = torch.cat([init_features, flow_features], dim=1)
        else:
            vf_features = init_features
        features = [vf_features]
        for name, layer in self.named_children():
            # print(name, layer)
            new_features = layer(*features)
            # features.append(new_features)
            features = [new_features]
        return [new_features, flow_features]
        # return [torch.cat(features, 1), flow_features]


class DenseNetUnit(nn.Module):
    def __init__(self, growth_rate=12, block_config=(8, 8, 8), compression=0.5, num_init_features=24,
                 bn_size=4, drop_rate=0.2, efficient=True, nb_flow=2,
                 channels=3, steps=1, height=16, width=8,
                 gcn_type='gcn', fusion=False, gcn_layer=3, flow_feature=True, gate_type=1, alpha=0.5, beta=0.5):
        super(DenseNetUnit, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
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

        # gnn = GCN if self.gcn_type.lower() == 'gcn' else GAT
        # self.gcn_dynamic = dict()
        # self.gcn_static = dict()
        # self.conv = dict()
        # for i in range(gcn_layer):
        #     self.gcn_dynamic[i] = gnn(self.nb_flow * self.channels if i == 0 else self.num_init_features,
        #                               self.num_init_features, self.num_init_features, dropout=0.5).to(device)
        #     self.gcn_static[i] = gnn(48 * 2 if i == 0 else self.num_init_features,
        #                              self.num_init_features, self.num_init_features, dropout=0.5).to(device)
        #     self.conv[i] = nn.Conv2d(self.nb_flow * self.channels if i == 0 else self.num_init_features,
        #                              self.num_init_features, kernel_size=3, padding=1).to(device)
        #     # self.conv[i] = conv_3x3_bn(self.nb_flow * self.channels if i == 0 else self.num_init_features,
        #     #                            self.num_init_features, kernel=3, padding=1).to(device)
        # self.dynamic_gating = Gating(self.num_init_features, self.num_init_features, bias=True).to(device)
        # self.static_gating = Gating(self.num_init_features, self.num_init_features, bias=True).to(device)

        if self.gcn_type.lower() == 'gcn':
            gcn_dynamic_1 = GCN(self.nb_flow * self.channels, self.num_init_features, self.num_init_features,
                                dropout=0.5)
            gcn_dynamic_2 = GCN(self.num_init_features, self.num_init_features, self.num_init_features,
                                dropout=0.5)
            gcn_dynamic_3 = GCN(self.num_init_features, self.num_init_features, self.num_init_features, dropout=0.5)
            gcn_dynamic_4 = GCN(self.num_init_features, self.num_init_features, self.num_init_features, dropout=0.5)
            gcn_static_1 = GCN(48 * 2, self.num_init_features, self.num_init_features,
                               dropout=0.5)
            gcn_static_2 = GCN(self.num_init_features, self.num_init_features, self.num_init_features,
                               dropout=0.5)
            gcn_static_3 = GCN(self.num_init_features, self.num_init_features, self.num_init_features, dropout=0.5)
            gcn_static_4 = GCN(self.num_init_features, self.num_init_features, self.num_init_features, dropout=0.5)
        else:
            gcn_dynamic_1 = GAT(self.nb_flow * self.channels, self.num_init_features, self.num_init_features,
                                dropout=0.5)
            gcn_dynamic_2 = GAT(self.num_init_features, self.num_init_features, self.num_init_features,
                                dropout=0.5)
            gcn_dynamic_3 = GAT(self.num_init_features, self.num_init_features, self.num_init_features, dropout=0.5)
            gcn_dynamic_4 = GAT(self.num_init_features, self.num_init_features, self.num_init_features, dropout=0.5)

            gcn_static_1 = GAT(48 * 2, self.num_init_features, self.num_init_features,
                               dropout=0.5)
            gcn_static_2 = GAT(self.num_init_features, self.num_init_features, self.num_init_features,
                               dropout=0.5)
            gcn_static_3 = GAT(self.num_init_features, self.num_init_features, self.num_init_features, dropout=0.5)
            gcn_static_4 = GAT(self.num_init_features, self.num_init_features, self.num_init_features, dropout=0.5)

        self.dynamic_1 = gcn_dynamic_1
        self.dynamic_2 = gcn_dynamic_2
        self.dynamic_3 = gcn_dynamic_3
        self.dynamic_4 = gcn_dynamic_4

        self.static_1 = gcn_static_1
        self.static_2 = gcn_static_2
        self.static_3 = gcn_static_3
        self.static_4 = gcn_static_4

        conv_1 = nn.Conv2d(self.nb_flow * self.channels, self.num_init_features, kernel_size=3, padding=1)
        conv_2 = nn.Conv2d(self.num_init_features, self.num_init_features, kernel_size=3, padding=1)
        conv_3 = nn.Conv2d(self.num_init_features, self.num_init_features, kernel_size=3, padding=1)
        conv_4 = nn.Conv2d(self.num_init_features, self.num_init_features, kernel_size=3, padding=1)

        self.conv_1 = conv_1
        self.conv_2 = conv_2
        self.conv_3 = conv_3
        self.conv_4 = conv_4

        self.dynamic_gating = Gating(self.num_init_features, self.num_init_features, bias=True)
        self.static_gating = Gating(self.num_init_features, self.num_init_features, bias=True)

        self.dense_input = nn.Conv2d(self.num_init_features, self.num_init_features, kernel_size=3, padding=1)
        # self.od_conv = nn.Conv2d(self.num_init_features, self.num_init_features, kernel_size=3, padding=1)

    def forward(self, x, adj=None, feature=None, road_adj=None, road_feature=None):
        x = x.view(-1, self.channels * self.nb_flow, self.height, self.width)
        if self.gcn_layer == 0:
            traffic_input = torch.relu(self.conv_1(x))  # if remove relu activation, the results improve
            flow_input = traffic_input
        else:
            if self.gcn_layer >= 1:
                traffic_out = self.conv_1(x)
                dynamic_out = self.dynamic_1(feature, adj)
                dynamic_gate = self.dynamic_gating(dynamic_out)
                # dynamic_gate = torch.sigmoid(dynamic_out)
                dynamic_gate = dynamic_gate.view(-1, self.height, self.width,
                                                 self.num_init_features).permute(0, 3, 1, 2)
                dynamic_gate = dynamic_gate.contiguous()
                static_out = self.static_1(road_feature, road_adj)
                static_gate = self.static_gating(static_out)
                # static_gate = torch.sigmoid(static_out)
                static_gate = static_gate.view(-1, self.height, self.width,
                                               self.num_init_features).permute(0, 3, 1, 2)
                static_gate = static_gate.contiguous()
                if self.gate_type == 1:
                    traffic_out = torch.relu(traffic_out * dynamic_gate)
                elif self.gate_type == 2:
                    traffic_out = torch.relu(traffic_out * static_gate)
                else:
                    traffic_out = torch.relu(traffic_out * (self.alpha * dynamic_gate + self.beta * static_gate))
            if self.gcn_layer >= 2:
                traffic_out = self.conv_2(traffic_out)
                dynamic_out = self.dynamic_2(torch.relu(dynamic_out), adj)
                dynamic_gate = self.dynamic_gating(dynamic_out)
                # dynamic_gate = torch.sigmoid(dynamic_out)
                dynamic_gate = dynamic_gate.view(-1, self.height, self.width,
                                                 self.num_init_features).permute(0, 3, 1, 2)
                dynamic_gate = dynamic_gate.contiguous()

                static_out = self.static_2(torch.relu(static_out), road_adj)
                static_gate = self.static_gating(static_out)
                # static_gate = torch.sigmoid(static_out)
                static_gate = static_gate.view(-1, self.height, self.width,
                                               self.num_init_features).permute(0, 3, 1, 2)
                static_gate = static_gate.contiguous()

                if self.gate_type == 1:
                    traffic_out = torch.relu(traffic_out * dynamic_gate)
                elif self.gate_type == 2:
                    traffic_out = torch.relu(traffic_out * static_gate)
                else:
                    traffic_out = torch.relu(traffic_out * (self.alpha * dynamic_gate + self.beta * static_gate))
            if self.gcn_layer >= 3:
                traffic_out = self.conv_3(traffic_out)
                dynamic_out = self.dynamic_3(torch.relu(dynamic_out), adj)
                dynamic_gate = self.dynamic_gating(dynamic_out)
                # dynamic_gate = torch.sigmoid(dynamic_out)
                dynamic_gate = dynamic_gate.view(-1, self.height, self.width,
                                                 self.num_init_features).permute(0, 3, 1, 2)
                dynamic_gate = dynamic_gate.contiguous()

                static_out = self.static_3(torch.relu(static_out), road_adj)
                static_gate = self.static_gating(static_out)
                # static_gate = torch.sigmoid(static_out)
                static_gate = static_gate.view(-1, self.height, self.width,
                                               self.num_init_features).permute(0, 3, 1, 2)
                static_gate = static_gate.contiguous()

                if self.gate_type == 1:
                    traffic_out = torch.relu(traffic_out * dynamic_gate)
                elif self.gate_type == 2:
                    traffic_out = torch.relu(traffic_out * static_gate)
                else:
                    traffic_out = torch.relu(traffic_out * (self.alpha * dynamic_gate + self.beta * static_gate))

            traffic_input = self.dense_input(traffic_out)

            if self.gate_type == 1:
                flow_input = dynamic_gate
            elif self.gate_type == 2:
                flow_input = static_gate
            elif self.gate_type == 3:
                flow_input = self.alpha * dynamic_gate + self.beta * static_gate
            else:
                flow_input = dynamic_gate

        # out, _ = self.features([traffic_input, flow_input])
        # out = self.last_component(out)
        return traffic_input, flow_input


class Gating(nn.Module):
    def __init__(self, in_size, out_size, bias=False):
        super(Gating, self).__init__()
        self.in_features = in_size
        self.out_features = out_size
        self.weight = Parameter(torch.FloatTensor(in_size, out_size))
        self.slope = Parameter(torch.rand(1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        out = torch.matmul(x, self.weight)
        if self.bias is not None:
            out += self.bias
        out[out < self.slope] = 1.
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Fusion(nn.Module):
    def __init__(self, c=2, h=16, w=8):
        super(Fusion, self).__init__()
        self.weight = Parameter(torch.ones(c, h, w))
        self.register_parameter('bias', None)

    def forward(self, inputs):
        return self.weight * inputs


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
        self.num_init_features = 24

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
        # self.features = EfficientNet.from_name('efficientnet-b0')

        self.features = nn.Sequential()
        growth_rate = 12
        compression = 0.5
        num_features = self.num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=4,
                growth_rate=growth_rate,
                drop_rate=0.2,
                efficient=True,
                num_init=self.num_init_features,
                flow_feature=flow_feature,
                gcn_layer=gcn_layer
            )
            self.features.add_module('dense_block_%d' % (i + 1), block)
            if flow_feature & (gcn_layer > 0):
                num_features = num_features + num_layers * growth_rate + self.num_init_features
            else:
                num_features = num_features + num_layers * growth_rate
            # num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _TransitionTP(num_input_features=num_features,
                                      num_output_features=int(num_features * compression))
                self.features.add_module('transition_%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm, add this layer will results in performance degradation
        self.last_component = nn.Sequential()
        self.last_component.add_module('last_norm', nn.BatchNorm2d(num_features))
        self.last_component.add_module('last_relu', nn.ReLU(inplace=True))
        self.last_component.add_module('last_conv', nn.Conv2d(num_features,
                                                              nb_flow * steps, kernel_size=3, padding=1))

    def forward(self, vx, adj=None, feature=None, road_net=None, road_feature=None):
        output = 0
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
                                                      self.close * self.nb_flow:(
                                                                                            self.close + self.period) * self.nb_flow],
                                                      road_net, road_feature)
            output += period_out
            flow += period_flow
        if self.trend > 0:
            trend_out, trend_flow = self.trend_net(vx[:, self.close + self.period:], adj[:, 2],
                                                   feature[:, 2, :, (self.close + self.period) * self.nb_flow:],
                                                   road_net, road_feature)
            output += trend_out
            flow += trend_flow

        output, _ = self.features([output, flow])
        # output = self.features(output)
        output = self.last_component(output)

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


class Discriminator(torch.nn.Module):
    """
    Design the D network
    """

    def __init__(self, steps=1, nb_flow=2):
        super(Discriminator, self).__init__()
        self.k_steps = steps
        self.nb_flows = nb_flow
        fms = 16
        kernel_size = 3
        leaky = 0.2
        self.main = nn.Sequential(
            # input size -1 * 2 * 10 * 20
            nn.Conv2d(steps * nb_flow, fms, kernel_size, 2, 1, bias=False),
            nn.LeakyReLU(leaky, inplace=True),

            # current size -1 * 16 * 5 * 10
            nn.Conv2d(fms, fms * 2, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(fms * 2),
            nn.LeakyReLU(leaky, inplace=True),

            # current size -1 * 32 * 3 * 5
            nn.Conv2d(fms * 2, fms * 4, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(fms * 4),
            nn.LeakyReLU(leaky, inplace=True),
            # current size -1 * 64 * 2 * 3
        )

        self.linear = nn.Sequential(
            nn.Linear(64 * 3 * 2, 64, bias=False),
            nn.LeakyReLU(leaky, inplace=True),
            nn.Linear(64, 32, bias=False),
            nn.LeakyReLU(leaky, inplace=True),
            nn.Linear(32, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        out = out.reshape(-1, 64 * 3 * 2)
        out = self.linear(out)
        return out
