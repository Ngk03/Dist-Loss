import os
import datetime
import argparse
import logging
from collections import Counter

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from net1d import Net1D
from torch.optim.lr_scheduler import StepLR
from dataset import MyDataset
from fast_soft_sort.pytorch_ops import soft_sort

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# model
parser.add_argument('--in_channels', type=int, default=1, help='The channel number of the input')
parser.add_argument('--base_filters', type=int, default=32, help='The number of base filters')
parser.add_argument('--ratio', type=int, default=1, help='The incremental ratio')
parser.add_argument('--filter_list', type=list, default=[32,64,128], help='The list of sequential filters')
parser.add_argument('--m_blocks_list', type=list, default=[2,2,2], help='The corresponding number of base filters')
parser.add_argument('--kernel_size', type=int, default=5, help='The conv kernel size')
parser.add_argument('--stride', type=int, default=2, help='The conv stride')
parser.add_argument('--groups_width', type=int, default=32, help='The width of conv groups')
parser.add_argument('--verbose', action='store_true', default=False, help='Whether to enable verbose mode')
parser.add_argument('--min_label', type=int, default=21, help='The minimum of labels')
parser.add_argument('--max_label', type=int, default=100, help='The maximum of labels')
parser.add_argument('--n_classes', type=int, default=1, help='The channel number of the output')

# hyperparameters
parser.add_argument('--lr', type=float, default=3e-3, help='The learning rate')
parser.add_argument('--batch_size', type=int, default=2048, help='The batch size')
parser.add_argument('--shuffle', action='store_false', default=True, help='Whether to shuffle the data')
parser.add_argument('--num_workers', type=int, default=16, help='The number of workers when data loading')
parser.add_argument('--drop_last', action='store_false', default=True, help='Whether to drop the last batch of data')
parser.add_argument('--epochs', type=int, default=100, help='The number of epochs for training')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='The type of optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='The weight for regularization')
parser.add_argument('--loss_weight', type=float, default=1.2, help='The weight for distribution-based loss')
parser.add_argument('--step_size', type=int, default=25, help='The step size for learning rate decay')
parser.add_argument('--gamma', type=float, default=0.1, help='The weight for learning rate decay')
parser.add_argument('--regularization_strength', type=float, default=0.1, help='The regularization strength for differentiable sorting')

# label distribution estimation
parser.add_argument('--bw_method', default=0.5, help='The method for determining bandwidth')

# experiment related
parser.add_argument('--train_data_dir', type=str, default='/home/nieguangkun/proj/ppg_age/data/train.npy', help='The dir of training data')
parser.add_argument('--valid_data_dir', type=str, default='/home/nieguangkun/proj/ppg_age/data/valid.npy', help='The dir of valid data')
parser.add_argument('--test_data_dir', type=str, default='/home/nieguangkun/proj/ppg_age/data/test.npy', help='The dir of tets data')
parser.add_argument('--store_root', type=str, default='/home/nieguangkun/ppg_age/exp/distribution_based/logging', help='Root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='', help='experiment store name')
parser.add_argument('--save_params_dir', type=str, default='/home/nieguangkun/ppg_age/params/distribution_based/net1d.pth', help='The save dir for model params')
parser.add_argument('--save_fig_dir', type=str, default='/home/nieguangkun/ppg_age/exp/distribution_based/results/demo.png', help='Path for saving test results')
parser.add_argument('--save_pred_dir', type=str, default='/home/nieguangkun/ppg_age/exp/distribution_based/results/pred.npz', help='Path for saving test results')

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()

if len(args.store_name):
    args.store_name = f'_{args.store_name}'
args.store_name += f'_lr_{args.lr}_batch_size_{args.batch_size}_epochs_{args.epochs}_loss_weight_{args.loss_weight}'

timestamp = str(datetime.datetime.now())
timestamp = '-'.join(timestamp.split(' '))
args.store_name += f'_{timestamp}'

prepare_folders(args)

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.store_name, 'training.log')),
        logging.StreamHandler()
    ])

def main():

    # Data
    print('===========================> Preparing data...')

    train_data = np.load(args.train_data_dir)
    train_ppg = normalization(train_data[:,1:])
    train_ppg = np.expand_dims(train_ppg, axis=1)
    train_age = train_data[:,0]

    valid_data = np.load(args.valid_data_dir)
    valid_ppg = normalization(valid_data[:,1:])
    valid_ppg = np.expand_dims(valid_ppg, axis=1)
    valid_age = valid_data[:,0]

    test_data = np.load(args.test_data_dir)
    test_ppg = normalization(test_data[:,1:])
    test_ppg = np.expand_dims(test_ppg, axis=1)
    test_age = test_data[:,0]

    density = get_ld(labels=train_age, bw_method=args.bw_method, min_label=args.min_label, max_label=args.max_label)
    batch_distribution_labels = get_batch_distribution_labels(density, args.batch_size, min_label=args.min_label).reshape(-1,1)
    batch_distribution_labels = torch.tensor(batch_distribution_labels).clone().type(torch.float32).cuda()

    train_ds = MyDataset(train_ppg, train_age, density)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last)

    valid_ds = MyDataset(valid_ppg, valid_age, density)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last)

    test_ds = MyDataset(test_ppg, test_age, None)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    print(f"Training data size: {len(train_ds)}")
    print(f"Validation data size: {len(valid_ds)}")
    print(f"Test data size: {len(test_ds)}")

    # Model
    print('===========================> Building model...')

    net = Net1D(
        in_channels=args.in_channels,
        base_filters=args.base_filters,
        ratio=args.ratio,
        filter_list=args.filter_list,
        m_blocks_list=args.m_blocks_list,
        kernel_size=args.kernel_size,
        stride=args.stride,
        groups_width=args.groups_width,
        verbose=args.verbose,
        min_label=args.min_label, 
        max_label=args.max_label,
        n_classes=args.n_classes
        )
    net = net.cuda()

    # Loss, optimizer, and scheduler
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay) if args.optimizer == 'Adam' else \
            optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    # Training
    print('===========================> Training model...')
    train(train_dl, valid_dl, net, optimizer, args.epochs, args.regularization_strength, loss_fn, args.loss_weight, batch_distribution_labels, args.save_params_dir, scheduler)
 
    # Testing
    print('===========================> Testing model...')
    test(test_dl, net, args.save_params_dir, args.save_fig_dir, args.save_pred_dir)

def train(train_dl, valid_dl, net, optimizer, epochs, regularization_strength, loss_fn, loss_weight, batch_distribution_labels, save_params_dir, scheduler):

    best_valid_loss = float('inf')

    for epoch in range(epochs):

        train_total_loss = 0
        train_Dist_loss = 0
        train_L1_loss = 0
        valid_total_loss = 0 
        valid_Dist_loss = 0 
        valid_L1_loss = 0 

        net.train()
        for batch_idx, (ppg, age, density) in enumerate(train_dl):
            ppg, age, density = ppg.cuda(), age.cuda(), density.cuda()
            out = net(ppg)
            sorted_out = soft_sort(out.reshape(1,-1).cpu(), regularization_strength=regularization_strength).reshape(-1,1).cuda()
            Dist_loss = loss_weight * loss_fn(batch_distribution_labels, sorted_out)
            L1_loss = loss_fn(age, out)
            total_loss = Dist_loss + L1_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_total_loss += total_loss
            train_Dist_loss += Dist_loss
            train_L1_loss += L1_loss
        train_total_loss = train_total_loss / (batch_idx + 1)
        train_Dist_loss = train_Dist_loss / (batch_idx + 1)
        train_L1_loss = train_L1_loss / (batch_idx + 1)

        net.eval()
        for batch_idx, (ppg, age, density) in enumerate(valid_dl):
            with torch.inference_mode():
                ppg, age, density = ppg.cuda(), age.cuda(), density.cuda()
                out = net(ppg)
                sorted_out = soft_sort(out.reshape(1,-1).cpu(), regularization_strength=regularization_strength).reshape(-1,1).cuda()
                Dist_loss = loss_weight * loss_fn(batch_distribution_labels, sorted_out)
                L1_loss = loss_fn(age, out)
                total_loss = Dist_loss + L1_loss
                valid_total_loss += total_loss
                valid_Dist_loss += Dist_loss
                valid_L1_loss += L1_loss
        valid_total_loss = valid_total_loss / (batch_idx + 1)
        valid_Dist_loss = valid_Dist_loss / (batch_idx + 1)
        valid_L1_loss = valid_L1_loss / (batch_idx + 1)
        
        if valid_total_loss <= best_valid_loss:
            torch.save(net.state_dict(), save_params_dir)
            best_valid_loss = valid_total_loss
        
        scheduler.step()

        print(f"Epoch [{epoch+1:2d}/{epochs}]", end=' ')
        print(f"train loss: {train_total_loss.item():.3f} = {train_Dist_loss.item():.3f} + {train_L1_loss.item():.3f}", end=' ')
        print('||', end=' ')
        print(f"valid loss: {valid_total_loss.item():.3f} = {valid_Dist_loss.item():.3f} + {valid_L1_loss.item():.3f}", end=' ')
        print('||', end=' ')
        print(f"lr: {optimizer.param_groups[0]['lr']}")

    print("Training finished.")

def test(test_dl, net, save_params_dir, save_fig_dir, save_pred_dir):

    pred, true = [], []

    params = torch.load(save_params_dir)
    net.load_state_dict(params)

    for ppg, age, _ in test_dl:
        with torch.inference_mode():
            ppg, age = ppg.cuda(), age.cuda()
            out = net(ppg)
            out = list(out.detach().cpu().numpy().squeeze())
            pred += out
            true += list(age.cpu().numpy().squeeze())
    pred = np.array(pred)
    true = np.array(true)
    np.savez(save_pred_dir, pred=pred, true=true)

    error = pred - true
    mean = error.mean()
    std = error.std()
    print(f'error mean: {mean:.3f}, error std: {std:.3f}')
    coef = np.corrcoef(pred, true)[0][1]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    error = pred - true
    mean = error.mean()
    std = error.std()
    axs[0, 0].scatter(true, pred, alpha=0.2, s=1, label=f'coef = {coef:.3f}')
    axs[0, 0].plot([21, 100], [21, 100], linestyle='--', linewidth=0.5, c='r')
    axs[0, 0].set_xlim([21, 100])
    axs[0, 0].set_ylim([21, 100])
    axs[0, 0].set_xlabel('True Values')
    axs[0, 0].set_ylabel('Predicted Values')
    axs[0, 0].set_title('Scatter Plot')
    axs[0, 0].legend()

    count_pred = Counter(pred.round())
    count_true = Counter(true)
    sorted_pred_keys, sorted_pred_values = zip(*sorted(zip(count_pred.keys(), count_pred.values())))
    sorted_true_keys, sorted_true_values = zip(*sorted(zip(count_true.keys(), count_true.values())))
    axs[0, 1].bar(sorted_true_keys, sorted_true_values, alpha=0.5, color='darkred', label='Actual distribution')
    axs[0, 1].bar(sorted_pred_keys, sorted_pred_values, alpha=0.5, label='Predicted distribution')
    axs[0, 1].set_xlabel('Values')
    axs[0, 1].set_ylabel('Counts')
    axs[0, 1].set_title('Bar Plot')
    axs[0, 1].legend()

    sns.violinplot(data=[pred, true], ax=axs[1, 0])
    axs[1, 0].set_title('Violin Plot')

    pred_percentiles = np.percentile(pred, np.linspace(0, 100, 100))
    true_percentiles = np.percentile(true, np.linspace(0, 100, 100))
    coef = np.corrcoef(pred_percentiles, true_percentiles)[0][1]
    axs[1, 1].plot(pred_percentiles, true_percentiles, marker='o', linestyle='', label=f'coef: {coef:.3f}')
    axs[1, 1].plot([30, 90], [30, 90], color='red', linestyle='--')  
    axs[1, 1].set_xlabel('Predicted Values Percentiles')
    axs[1, 1].set_ylabel('True Values Percentiles')
    axs[1, 1].set_title('QQ Plot')
    axs[1, 1].legend()

    plt.tight_layout()

    plt.savefig(save_fig_dir, dpi=600)

    plt.show()

if __name__ == '__main__':
    main()