# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: st_datasets.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-08-11 (YYYY-MM-DD)
-----------------------------------------------
"""
import numpy as np
import scipy.sparse as sp


class MinMaxNormalization11(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self.min = X.min()
        self.max = X.max()
        # print("min:", self.min, "max:", self.max)

    def transform(self, X):
        X = 1. * (X - self.min) / (self.max - self.min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self.max - self.min) + self.min
        return X


class MinMaxNormalization01(object):
    '''MinMax Normalization --> [0, 1]
       x = (x - min) / (max - min).
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self.min = X.min()
        self.max = X.max()
        # print("min:", self.min, "max:", self.max)

    def transform(self, X):
        X = 1. * (X - self.min) / (self.max - self.min)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = 1. * X * (self.max - self.min) + self.min
        return X


class ZScore(object):
    def __init__(self):
        pass

    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

    def transform(self, x):
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        return x * self.std + self.mean


def generate_data(data, c, p, t, start_idx, steps):
    X, y = [], []
    X_sum = []

    for idx in range(start_idx, len(data) - steps + 1):
        features = []
        features_sum = []
        target = []
        if c > 0:
            xc_ = [data[idx - i] for i in range(1, c + 1)]
            # xc_ has c elements, each with size of (c, nb_flow, height, width)
            # we stack xc_ along the first axis, thus the data size now is (c*nb_flow, height, width)
            features.append(xc_)
            features_sum.append(np.array(xc_).sum(axis=0))
        if p > 0:
            xp_ = [data[idx - j * 48] for j in range(1, p + 1)]
            features.append(xp_)
            features_sum.append(np.array(xp_).sum(axis=0))
        if t > 0:
            xt_ = [data[idx - k * 48 * 7] for k in range(1, t + 1)]
            features.append(xt_)
            features_sum.append(np.array(xt_).sum(axis=0))

        if steps >= 1:
            target = [data[idx + n] for n in range(steps)]
            target = np.vstack(target)

        features = np.concatenate(features, axis=0)
        X.append(features)
        X_sum.append(features_sum)
        y.append(target)

    X = np.array(X)
    y = np.array(y)
    X_sum = np.array(X_sum)
    return X, X_sum, y


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_adj(data):
    # data is with shape (nb_flow, N, N), so we first perform addition over the first axis, to make
    # the data is (N, N), that is, the adjacent matrix of OD graph
    adj = sp.coo_matrix(data.sum(axis=0))
    # next, make the graph symmetric, as this is claimed in kipf's paper and code
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # make the graph self loop
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    return adj.todense()


def get_feature(vm, channel):
    # vm is with shape (nb_sequences, nb_flow, h, w), we first reshape its shape to (nb_sequences*nb_flow, h*w)
    # that is, each row denotes the traffic feature of that region (node n)
    vx = vm.reshape((channel, -1)) + 0.1  # add a small term to avoid zero divide
    vx = sp.csr_matrix(np.moveaxis(vx, 0, 1))
    vx = normalize(vx)
    return vx.todense()


def load_traffic_data_v4(args, volume, flow, road_net=None):
    trend_days = 7
    period_days = 1
    samples_per_day = 48
    start_idx = max(args.close_size, args.period_size * period_days * samples_per_day,
                    args.trend_size * trend_days * samples_per_day)
    if args.norm_type == '01_sigmoid':
        mmn_volume = MinMaxNormalization01()
    elif args.norm_type == '11_tanh':
        mmn_volume = MinMaxNormalization11()
    elif args.norm_type == 'z_score_linear':
        mmn_volume = ZScore()
    elif args.norm_type == '01_tanh':
        mmn_volume = MinMaxNormalization01()
    else:
        raise Exception('Wrong Choice')
    # the test length is exactly the same as the length in STDN

    test_len = args.test_size
    mmn_volume.fit(volume[:-test_len])
    volume = mmn_volume.transform(volume)

    volume_x, volume_x_sum, volume_y = generate_data(volume, args.close_size, args.period_size,
                                                     args.trend_size, start_idx, args.steps)
    flow_x, flow_x_sum, flow_y = generate_data(flow, args.close_size, args.period_size, args.trend_size,
                                               start_idx, args.steps)

    n_sample, n_channel, n_flow, h, w = volume_x.shape
    new_flow = np.zeros(shape=(flow_x.shape[0], 3, h * w, h * w))
    feature = np.zeros(shape=(n_sample, 3, h * w, n_channel*n_flow))

    feat_len = 48
    # road_feat = volume[:feat_len]
    road_feat = get_feature(volume[:feat_len], feat_len*2)
    road_feat = road_feat[np.newaxis, :, :]
    road = get_adj(road_net)
    road = road[np.newaxis, :, :]

    # print('Building symmetric adjacency matrix')
    for t in range(flow_x.shape[0]):
        # if t > 100:
        #     break
        if args.close_size > 0:
            n_c = args.close_size * n_flow
            new_flow[t, 0] = get_adj(flow_x_sum[t, 0])
            feature[t, 0, :, :n_c] = get_feature(volume_x[t, 0:args.close_size], n_c)
        if args.period_size > 0:
            n_p = args.period_size * n_flow
            new_flow[t, 1] = get_adj(flow_x_sum[t, 1])
            feature[t, 1, :, n_c:n_c+n_p] = \
                get_feature(volume_x[t, args.close_size:args.close_size+args.period_size], n_p)
        if args.trend_size > 0:
            n_t = args.trend_size * n_flow
            new_flow[t, 2] = get_adj(flow_x_sum[t, 2])
            feature[t, 2, :, n_c+n_p:] = \
                get_feature(volume_x[t, args.close_size+args.period_size:], n_t)

    vx_train, vx_test = volume_x[:-test_len], volume_x[-test_len:]
    vy_train, vy_test = volume_y[:-test_len], volume_y[-test_len:]
    fx_train, fx_test = new_flow[:-test_len], new_flow[-test_len:]
    feature_train, feature_test = feature[:-test_len], feature[-test_len:]
    # fy_train, fy_test = flow_y[:-test_len], flow_y[-test_len:]
    return [vx_train, fx_train, feature_train, vy_train], [vx_test, fx_test, feature_test, vy_test], road, road_feat, mmn_volume
