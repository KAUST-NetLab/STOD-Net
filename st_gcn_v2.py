# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: st_gcn.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-08-12 (YYYY-MM-DD)
-----------------------------------------------
"""
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn import metrics

sys.path.append('../')
from st_dense_gcn.utils.st_datasets import load_traffic_data_v4
# from st_dense_gcn.utils.st_models import Generator
# from st_dense_gcn.utils.st_efficient import Generator
from st_dense_gcn.utils.st_models_v2 import Generator
# from st_dense_gcn.utils.st_models_v3 import Generator
import argparse

torch.manual_seed(89)

parse = argparse.ArgumentParser()
parse.add_argument('-traffic', type=str, default='taxi')

parse.add_argument('-close_size', type=int, default=5)
parse.add_argument('-period_size', type=int, default=0)
parse.add_argument('-trend_size', type=int, default=0)
parse.add_argument('-steps', type=int, default=1)
parse.add_argument('-test_size', type=int, default=480)
parse.add_argument('-val_ratio', type=int, default=0.1)
parse.add_argument('-nb_flow', type=int, default=2)

parse.add_argument('-gcn_layer', type=int, default=2)
parse.add_argument('-gate_type', type=int, default=3)
parse.add_argument('-alpha', type=float, default=0.5)
parse.add_argument('-beta', type=float, default=0.5)

parse.add_argument('-fusion', dest='fusion', action='store_true')
parse.add_argument('-no-fusion', dest='fusion', action='store_false')
parse.set_defaults(fusion=False)

parse.add_argument('-ibex', dest='ibex', action='store_true')
parse.add_argument('-no-ibex', dest='ibex', action='store_false')
parse.set_defaults(ibex=False)

parse.add_argument('-gcn_type', type=str, default='gat', help='gcn | gat')

parse.add_argument('-transfer', dest='transfer', action='store_true')
parse.add_argument('-no-transfer', dest='transfer', action='store_false')
parse.set_defaults(transfer=False)

parse.add_argument('-ff', dest='ff', action='store_true', help='flow feature')
parse.add_argument('-no-ff', dest='ff', action='store_false')
parse.set_defaults(ff=True)

parse.add_argument('-train', dest='train', action='store_true')
parse.add_argument('-no-train', dest='train', action='store_false')
parse.set_defaults(train=True)
parse.add_argument('-loss', type=str, default='l2', help='l1 | l2')
parse.add_argument('-norm_type', type=str, default='01_sigmoid', help='01_sigmoid | 01_tanh | 11_tanh | z_score_linear')
parse.add_argument('-lr', type=float, default=1e-4)
parse.add_argument('-depth', type=int, default=28, help='depth of st-densenet')
parse.add_argument('-batch_size', type=int, default=32, help='batch size')
parse.add_argument('-epoch_size', type=int, default=500, help='epochs')
parse.add_argument('-exp', type=int, default=0, help='number of exp')

args = parse.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, loader, optimizer, criterion, epoch_size, n_epoch=1, scheduler=None, road_adj=None,
                road_att=None):
    print_freq = 5

    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    road_net = torch.from_numpy(road_adj).float().to(device)
    road_feature = torch.from_numpy(road_att).float().to(device)

    # make the model on train mode
    model.train()

    start = time.time()

    for batch_idx, (volume, adj, attributes, target) in enumerate(loader):
        volume = volume.float().to(device)
        adj = adj.float().to(device)
        attributes = attributes.float().to(device)
        target = target.float().to(device)
        optimizer.zero_grad()

        output = model(volume, adj, attributes, road_net, road_feature)
        loss_1 = criterion[0](target, output)
        # RMSE Loss
        # https://discuss.pytorch.org/t/rmse-loss-function/16540/3
        loss_2 = torch.sqrt(criterion[1](target, output) + 1e-6)
        if args.loss.lower() == 'l1':
            loss = loss_1
        elif args.loss.lower() == 'l2':
            loss = loss_2
        elif args.loss.lower() == 'l1l2':
            loss = loss_1 + loss_2
        else:
            loss = loss_1

        # measure accuracy and record loss
        n_elements = torch.numel(output)
        pred = output.data.cpu()
        error.update(torch.sqrt(
            torch.mean(
                (pred - target.cpu()) ** 2
            )
        ), n_elements)

        losses.update(loss.item(), n_elements)

        # compute gradient and perform SGD one step

        loss.backward()
        optimizer.step()
        scheduler.step()

        # compute elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

        # Output training stats
        if batch_idx % print_freq == 0:
            print('Training [{:03d}/{:03d}][{:03d}/{:03d}]\tTime {:.3f} ({:.3f})\tLoss '
                  '{:.6f} ({:.6f})\tError {:.5f} ({:.5f})'.format(n_epoch + 1, epoch_size,
                                                                  batch_idx + 1,
                                                                  len(loader), batch_time.val, batch_time.avg,
                                                                  losses.val, losses.avg, error.val, error.avg),
                  flush=True)
    # return summarized statistics
    return batch_time.avg, losses.avg, error.avg


def test_epoch(model, loader, criterion, epoch_size, n_epoch=1, threshold=10., road_adj=None, road_att=None):
    model.eval()
    print_freq = 5

    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    road_net = torch.from_numpy(road_adj).float().to(device)
    road_feature = torch.from_numpy(road_att).float().to(device)

    start = time.time()

    for batch_idx, (volume, adj, attributes, target) in enumerate(loader):
        volume = volume.float().to(device)
        adj = adj.float().to(device)
        attributes = attributes.float().to(device)
        target = target.float().to(device)

        output = model(volume, adj, attributes, road_net, road_feature)
        loss_1 = criterion[0](target, output)
        loss_2 = torch.sqrt(criterion[1](target, output) + 1e-6)
        if args.loss.lower() == 'l1':
            loss = loss_1
        elif args.loss.lower() == 'l2':
            loss = loss_2
        elif args.loss.lower() == 'l1l2':
            loss = loss_1 + loss_2
        else:
            loss = loss_1

        # measure accuracy and record loss
        n_elements = torch.numel(output)
        pred = output.data.cpu()
        error.update(torch.sqrt(
            torch.mean(
                (pred - target.cpu()) ** 2
            )
        ), n_elements)

        losses.update(loss.item(), n_elements)

        # compute elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

        # Output training stats
        if batch_idx % print_freq == 0:
            print('Valid: [{:03d}/{:03d}][{:03d}/{:03d}]\tTime {:.3f} ({:.3f})\tLoss '
                  '{:.5f} ({:.5f})\tError {:.5f} ({:.5f})'.format(n_epoch + 1, epoch_size,
                                                                  batch_idx + 1,
                                                                  len(loader), batch_time.val, batch_time.avg,
                                                                  losses.val, losses.avg, error.val, error.avg),
                  flush=True)
    # return summarized statistics
    return batch_time.avg, losses.avg, error.avg


def predict(model, loader, mmn, road_adj=None, road_att=None):
    predictions, truth = [], []
    road_net = torch.from_numpy(road_adj).float().to(device)
    road_feature = torch.from_numpy(road_att).float().to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (volume, adj, attributes, label) in enumerate(loader):
            volume = volume.float().to(device)
            # print(volume.size())
            adj = adj.float().to(device)
            attributes = attributes.float().to(device)
            label = label.float().to(device)

            pred = model(volume, adj, attributes, road_net, road_feature)
            predictions.append(pred)
            truth.append(label)
    truth_all = mmn.inverse_transform(torch.cat(truth, dim=0).cpu().numpy())
    pred_all = mmn.inverse_transform(torch.cat(predictions, dim=0).cpu().numpy())
    return pred_all, truth_all


def eval_metric(y, pred_y, threshold):
    pickup_y = y[:, 0]
    dropoff_y = y[:, 1]
    pickup_pred_y = pred_y[:, 0]
    dropoff_pred_y = pred_y[:, 1]
    pickup_mask = pickup_y > threshold
    dropoff_mask = dropoff_y > threshold
    # pickup part
    if np.sum(pickup_mask) != 0:
        avg_pickup_mape = np.mean(
            np.abs(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask]) / pickup_y[pickup_mask])
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask])))
        avg_pickup_mae = np.mean(np.abs(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask]))
        avg_pickup_r2 = metrics.r2_score(pickup_y[pickup_mask], pickup_pred_y[pickup_mask])
    # dropoff part
    if np.sum(dropoff_mask) != 0:
        avg_dropoff_mape = np.mean(
            np.abs(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask]) / dropoff_y[dropoff_mask])
        avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask])))
        avg_dropoff_mae = np.mean(np.abs(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask]))
        avg_dropoff_r2 = metrics.r2_score(dropoff_y[dropoff_mask], dropoff_pred_y[dropoff_mask])

    return (avg_pickup_rmse, avg_pickup_mae, avg_pickup_r2, avg_pickup_mape), \
           (avg_dropoff_rmse, avg_dropoff_mae, avg_dropoff_r2, avg_dropoff_mape)


def eval_metric_no_mask(y, pred_y):
    pickup_y = y[:, 0]
    dropoff_y = y[:, 1]
    pickup_pred_y = pred_y[:, 0]
    dropoff_pred_y = pred_y[:, 1]

    pickup_mape_mask = pickup_y > 0
    dropoff_mape_mask = dropoff_y > 0
    # pickup part
    avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y - pickup_pred_y)))
    avg_pickup_mae = np.mean(np.abs(pickup_y - pickup_pred_y))
    avg_pickup_r2 = metrics.r2_score(pickup_y.ravel(), pickup_pred_y.ravel())
    avg_pickup_mape = np.mean(np.abs((pickup_y[pickup_mape_mask] -
                                      pickup_pred_y[pickup_mape_mask]) / pickup_y[pickup_mape_mask]))
    # dropoff part
    avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y - dropoff_pred_y)))
    avg_dropoff_mae = np.mean(np.abs(dropoff_y - dropoff_pred_y))
    avg_dropoff_r2 = metrics.r2_score(dropoff_y.ravel(), dropoff_pred_y.ravel())
    avg_dropoff_mape = np.mean(np.abs((dropoff_y[dropoff_mape_mask] -
                                       dropoff_pred_y[dropoff_mape_mask]) / dropoff_y[dropoff_mape_mask]))

    return (avg_pickup_rmse, avg_pickup_mae, avg_pickup_r2, avg_pickup_mape), \
           (avg_dropoff_rmse, avg_dropoff_mae, avg_dropoff_r2, avg_dropoff_mape)


def eval_metric_all_no_mask(y, pred_y):
    # all
    avg_rmse = np.sqrt(np.mean(np.square(y - pred_y)))
    avg_mae = np.mean(np.abs(y - pred_y))
    avg_r2 = metrics.r2_score(y.ravel(), pred_y.ravel())

    mape_mask = y > 0
    avg_mape = np.mean(np.abs(y[mape_mask] - pred_y[mape_mask]) / y[mape_mask])

    return avg_rmse, avg_mae, avg_r2, avg_mape


def eval_metric_all(y, pred_y, threshold):
    y_mask = y > threshold
    # pickup part
    if np.sum(y_mask) != 0:
        avg_mape = np.mean(
            np.abs(y[y_mask] - pred_y[y_mask]) / y[y_mask])
        avg_rmse = np.sqrt(np.mean(np.square(y[y_mask] - pred_y[y_mask])))
        avg_mae = np.mean(np.abs(y[y_mask] - pred_y[y_mask]))
        avg_r2 = metrics.r2_score(y[y_mask], pred_y[y_mask])

    return avg_rmse, avg_mae, avg_r2, avg_mape


def main():
    # main function
    if args.ibex:
        path = './results/data/'
    else:
        path = 'D:/Projects/Github/STDN/data/'
    if args.traffic == 'taxi':
        volume_train = np.load(open(path + 'volume_train.npz', 'rb'))['volume']
        volume_test = np.load(open(path + 'volume_test.npz', 'rb'))['volume']
        flow_train = np.load(open(path + 'flow_train.npz', 'rb'))['flow']
        flow_test = np.load(open(path + 'flow_test.npz', 'rb'))['flow']
        # road_net = np.sum(flow_train, axis=1).reshape((2, 200, 200))
        road_net = np.load(path + 'nyc_road.npy').reshape((1, 200, 200))
    elif args.traffic == 'bike':
        volume_train = np.load(open(path + 'bike_volume_train.npz', 'rb'))['volume']
        volume_test = np.load(open(path + 'bike_volume_test.npz', 'rb'))['volume']
        flow_train = np.load(open(path + 'bike_flow_train.npz', 'rb'))['flow']
        flow_test = np.load(open(path + 'bike_flow_test.npz', 'rb'))['flow']
        # road_net = np.sum(flow_train, axis=1).reshape((2, 200, 200))
        road_net = np.load(path + 'nyc_road.npy').reshape((1, 200, 200))
    elif args.traffic == 'chengdu':
        volume_train = np.load(open(path + 'chengdu_volume_train.npz', 'rb'))['volume']
        volume_test = np.load(open(path + 'chengdu_volume_test.npz', 'rb'))['volume']
        flow_train = np.load(open(path + 'chengdu_flow_train.npz', 'rb'))['flow']
        flow_test = np.load(open(path + 'chengdu_flow_test.npz', 'rb'))['flow']
        road_net = np.sum(flow_train, axis=1).reshape((2, 400, 400))
        # road_file = pd.read_csv(path + 'chengdu_road_edges_geo_20_20.csv', header=0)
        # road_net = np.zeros(shape=(2, 400, 400))
        # road_net[:, road_file['src'].values, road_file['dst'].values] = road_file['connectivity'].values
    elif args.traffic == 'xian':
        volume_train = np.load(open(path + 'xian_volume_train.npz', 'rb'))['volume']
        volume_test = np.load(open(path + 'xian_volume_test_.npz', 'rb'))['volume']
        flow_train = np.load(open(path + 'xian_flow_train.npz', 'rb'))['flow']
        flow_test = np.load(open(path + 'xian_flow_test.npz', 'rb'))['flow']
        road_net = np.sum(flow_train, axis=1).reshape((2, 400, 400))
        # road_file = pd.read_csv(path + 'xian_road_edges_geo_20_20.csv', header=0)
        # road_net = np.zeros(shape=(2, 400, 400))
        # road_net[:, road_file['src'].values, road_file['dst'].values] = road_file['connectivity'].values
    else:
        print('Wrong dataset name.')

    volume = np.concatenate([volume_train, volume_test])
    flow = np.concatenate([flow_train, flow_test], axis=1)
    sequences, h, w, flows = volume_train.shape
    volume = np.moveaxis(volume, -1, 1)
    flow = np.moveaxis(flow, 0, 1)
    flow = np.reshape(flow, [-1, flows, h * w, h * w])
    # print(block_flow.shape, block_volume.shape)
    train, test, road_adj, road_feat, mmn = load_traffic_data_v4(args, volume, flow, road_net)

    file_name = 'File_{:}_GCN_{:}_Layer_{:}_Fusion_{:}_LR_{:.5f}_Depth_{:}_Flow_{:}' \
                '_Loss_{:}_Gate_{:}_Alpha_{:}_Beta_{:}_c_{:}_p_{:}_t_{:}_Exp_{:}_epoch_{:}'.format(args.traffic,
                                                                                                   args.gcn_type,
                                                                                                   args.gcn_layer,
                                                                                                   args.fusion,
                                                                                                   args.lr,
                                                                                                   args.depth, args.ff,
                                                                                                   args.loss,
                                                                                                   args.gate_type,
                                                                                                   args.alpha,
                                                                                                   args.beta,
                                                                                                   args.close_size,
                                                                                                   args.period_size,
                                                                                                   args.trend_size,
                                                                                                   args.exp,
                                                                                                   args.epoch_size)

    n_samples, n_sequences, n_flow, h, w = train[0].shape
    train_dataset = [(volume_x, flow_x, feature_x, volume_y) for volume_x, flow_x, feature_x, volume_y
                     in zip(*train)]
    test_dataset = [(volume_x, flow_x, feature_x, volume_y) for volume_x, flow_x, feature_x, volume_y
                    in zip(*test)]
    val_dataset = train_dataset[-int(len(train_dataset) * args.val_ratio):] if args.val_ratio > 0 else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=True, pin_memory=True) if val_dataset is not None else None

    if (args.depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(args.depth - 4) // 6 for _ in range(3)]
    model = Generator(block_config=block_config, nb_flow=args.nb_flow,
                      channels=[args.close_size, args.period_size, args.trend_size], steps=args.steps, height=h,
                      width=w, gcn_type=args.gcn_type,
                      fusion=args.fusion, norm=args.norm_type, gcn_layer=args.gcn_layer,
                      flow_feature=args.ff).to(device)

    # ip = iter(train_loader)
    # from torchsummary.torchsummary import summary
    # from torchsummary import summary
    # summary(model, [(5, 2, 10, 20), (3, 200, 200), (3, 200, 10), (200, 200), (200, 96)])
    # print(model)

    criterion_l1 = torch.nn.L1Loss().to(device)
    criterion_l2 = torch.nn.MSELoss().to(device)
    criterion = [criterion_l1, criterion_l2]

    best_val_loss = 10.
    train_loss_hist = []
    val_loss_hist = []

    # we train our model n times and each time with 100 epochs
    for i in range(1):
        print('Training for the {:}-th time'.format(i + 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr * 10,
                                                        steps_per_epoch=len(train_loader), epochs=args.epoch_size)
        # # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=len(train_loader))
        if i > 0:
            checkpoint = torch.load(os.path.join('checkpoint', file_name + '_best_model.pt'))
            model.load_state_dict(checkpoint['net'])
        for epoch in range(args.epoch_size):
            _, train_loss, train_error = train_epoch(model, train_loader, optimizer, criterion, args.epoch_size,
                                                     n_epoch=epoch, scheduler=scheduler,
                                                     road_adj=road_adj, road_att=road_feat)
            _, val_loss, val_error = test_epoch(model, val_loader, criterion, args.epoch_size,
                                                n_epoch=epoch, threshold=10. / mmn.max,
                                                road_adj=road_adj, road_att=road_feat)

            pred, truth = predict(model, test_loader, mmn, road_adj=road_adj, road_att=road_feat)

            # results with mask
            (prmse_mask, pmae_mask, pr2_mask, pmape_mask), \
            (drmse_mask, dmae_mask, dr2_mask, dmape_mask) = eval_metric(truth, pred, 10.)
            rmse_mask, mae_mask, r2_mask, mape_mask = eval_metric_all(truth, pred, 10.)
            print('Round, {:}, Epoch, {:}, File, {:}, Pickup RMSE, {:.4f}, MAE, {:.4f}, MAPE, {:.5f}, '
                  'Dropoff RMSE, {:.4f}, MAE, {:.4f} MAPE: {:.5f}'.format(i + 1, epoch + 1, args.traffic,
                                                                          prmse_mask, pmae_mask, pmape_mask, drmse_mask,
                                                                          dmae_mask, dmape_mask), flush=True)

            # Save checkpoint.
            if val_loss < best_val_loss:
                # print('Saving..')
                state = {
                    'net': model.state_dict(),
                    'loss': val_loss,
                    'epoch': epoch,
                    'pred': pred,
                    'truth': truth,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, os.path.join('checkpoint', file_name + '_best_model.pt'))
                best_val_loss = val_loss

            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)

            if not os.path.isdir('curve'):
                os.mkdir('curve')
            torch.save({'train_loss': train_loss_hist, 'val_loss': val_loss_hist},
                       os.path.join('curve', file_name + '.curve'))

    checkpoint = torch.load(os.path.join('checkpoint', file_name + '_best_model.pt'))
    model.load_state_dict(checkpoint['net'])
    pred, truth = predict(model, test_loader, mmn, road_adj=road_adj, road_att=road_feat)

    # results with mask
    (prmse_mask, pmae_mask, pr2_mask, pmape_mask), \
    (drmse_mask, dmae_mask, dr2_mask, dmape_mask) = eval_metric(truth, pred, 10.)
    rmse_mask, mae_mask, r2_mask, mape_mask = eval_metric_all(truth, pred, 10.)

    # results without mask
    (prmse, pmae, pr2, pmape), (drmse, dmae, dr2, dmape) = eval_metric_no_mask(truth, pred)
    rmse, mae, r2, mape = eval_metric_all_no_mask(truth, pred)

    print(
        'Final --> File: {:}, GCN: {:}, Layer: {:}, Depth: {:}, LR: {:}, Exp: {:}, Loss: {:}, Gate: {:} , Alpha: {:}, Beta: {:}, '
        'C: {:}, P: {:}, T: {:}, Flow: {:}, Fusion: {:}, Epoch: {:},'
        'Pickup RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.5f}, '
        'Dropoff RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.5f}, RMSE: {:.4f}, '
        'MAE: {:.4f}, MAPE: {:.5f}'.format(args.traffic,
                                           args.gcn_type,
                                           args.gcn_layer, args.depth,
                                           args.lr,
                                           args.exp,
                                           args.loss, args.gate_type, args.alpha, args.beta,
                                           args.close_size,
                                           args.period_size,
                                           args.trend_size,
                                           args.ff,
                                           args.fusion, args.epoch_size,
                                           prmse_mask,
                                           pmae_mask,
                                           pmape_mask,
                                           drmse_mask,
                                           dmae_mask,
                                           dmape_mask,
                                           rmse, mae, mape),
        flush=True)
    print('STOD With-mask-->prmse: {:.4f}, pmae: {:.4f}, pr2: {:.4f}, pmape: {:.4f}; drmse: {:.4f}, dmae: {:.4f}'
          ' dr2: {:.4f}, dmape: {:.4f}, rmse: {:.4f}, '
          'mae: {:.4f}, r2: {:.4f}, mape: {:.4f}'.format(prmse_mask, pmae_mask, pr2_mask, pmape_mask, drmse_mask,
                                                         dmae_mask, dr2_mask, dmape_mask, rmse_mask, mae_mask,
                                                         r2_mask, mape_mask))
    print('STOD Without-mask-->prmse: {:.4f}, pmae: {:.4f}, pr2: {:.4f}, pmape: {:.4f}; drmse: {:.4f}, dmae: {:.4f}'
          ' dr2: {:.4f}, dmape: {:.4f}, rmse: {:.4f}, '
          'mae: {:.4f}, r2: {:.4f}, mape: {:.4f}'.format(prmse, pmae, pr2, pmape, drmse,
                                                         dmae, dr2, dmape, rmse, mae,
                                                         r2, mape))
    # np.savez(file_name+'.npz', truth=truth, pred=pred)


if __name__ == '__main__':
    main()
