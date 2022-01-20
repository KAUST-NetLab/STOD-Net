# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: base_models.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-08-23 (YYYY-MM-DD)
-----------------------------------------------
"""
import torch
from torch import nn
from torch.autograd import Variable
import sys

sys.path.append('../')
from st_dense_gcn.utils.st_models import ConvLSTMLayer


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        batch_size, seq, flow, h, w = x.shape
        out = torch.relu(self.fc1(x.reshape(batch_size, -1)))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out).view(batch_size, flow, h, w)
        return out


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        batch_size, seq, flow, h, w = x.shape
        out = self.fc1(x.reshape(batch_size, -1))
        out = out.reshape((batch_size, flow, h, w))
        return out


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.input_dim = 400
        self.hidden_dim = 64
        self.out_dim = 400
        self.num_layers = 2
        self.h = 10
        self.w = 20
        self.close_size = args.close_size
        self.period_size = args.period_size
        self.trend_size = args.trend_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lstm_close = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, dropout=0.2)
        self.lstm_period = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                   batch_first=True, dropout=0.2)
        self.lstm_trend = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, dropout=0.2)
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, dropout=0.2)

        self.linear_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x):
        bz = x.size(0)
        h0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim)).to(self.device)
        c0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim)).to(self.device)

        self.lstm_close.flatten_parameters()
        self.lstm_period.flatten_parameters()
        self.lstm_trend.flatten_parameters()
        self.lstm.flatten_parameters()

        # xc = x[:, :self.close_size]
        # xp = x[:, self.close_size:self.close_size + self.period_size]
        # xt = x[:, self.close_size + self.period_size:]

        x_out, _ = self.lstm(x[:, :self.close_size].reshape(bz, self.close_size, -1), (h0, c0))

        # xc_out, _ = self.lstm_close(xc.reshape(bz, self.close_size, -1), (h0, c0))
        # xp_out, _ = self.lstm_period(xp.reshape(bz, self.period_size, -1), (h0, c0))
        # xt_out, _ = self.lstm_trend(xt.reshape(bz, self.trend_size, -1), (h0, c0))

        # out = xc_out[:, -1] + xp_out[:, -1] + xt_out[:, -1]
        out = x_out[:, -1]
        y_pred = self.linear_layer(out).view(bz, 2, self.h, self.w)
        # y_pred = y_pred
        return y_pred


class ConvLSTM(nn.Module):
    def __init__(self, channels, height, width):
        super(ConvLSTM, self).__init__()
        self.close = channels[0]
        self.period = channels[1]
        self.trend = channels[2]

        self.conv_close = ConvLSTMLayer(24, 3, [-1, channels[0], 2, height, width])
        self.conv_period = ConvLSTMLayer(24, 3, [-1, channels[1], 2, height, width])
        self.conv_trend = ConvLSTMLayer(24, 3, [-1, channels[2], 2, height, width])

        self.conv = ConvLSTMLayer(24, 3, [-1, self.close+self.period+self.trend, 2, height, width])
        self.last_conv_1 = nn.Conv2d(24, 2, kernel_size=3, padding=1)

    def forward(self, x):
        # close_out = torch.relu(self.conv_close(x[:, :self.close]))
        # period_out = torch.relu(self.conv_period(x[:, self.close:self.close + self.period]))
        # trend_out = torch.relu(self.conv_trend(x[:, self.close + self.period:]))
        out = self.conv(x)
        # out = torch.cat([close_out, period_out, trend_out], dim=1)
        # out = torch.relu(self.last_conv_1(out))
        out = self.last_conv_1(out)
        return out
