import argparse
import random
import time
import torch
import os
import shutil
import numpy as np
from tensorboard_logger import tensorboard_logger
from src.network import GNet
from src.trainer import Trainer
from src.utils.data_loader import FileLoader, FileLoaderNew

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=42, help='seed')
    parser.add_argument('-data', default='twitter', help='data folder name')
    parser.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    parser.add_argument('-num_epochs', type=int, default=100, help='epochs')
    parser.add_argument('-batch', type=int, default=2048, help='batch size')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-deg_as_tag', type=int, default=0, help='1 or degree')
    parser.add_argument('-l_num', type=int, default=3, help='layer num')
    parser.add_argument('-h_dim', type=int, default=128, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=128, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0.3, help='drop net')
    parser.add_argument('-drop_c', type=float, default=0.3, help='drop output')
    parser.add_argument('-act_n', type=str, default='ELU', help='network act')
    parser.add_argument('-act_c', type=str, default='ELU', help='output act')
    parser.add_argument('-ks', nargs='+', type=str, default="0.9 0.8 0.7")
    parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    args, _ = parser.parse_known_args()
    args.ks = [0.9, 0.8, 0.7]
    return args


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def app_run(args, G_data, fold_idx):
    G_data.use_fold_data(fold_idx)
    net = GNet(G_data.feat_dim, G_data.num_class, args)
    trainer = Trainer(args, net, G_data)
    trainer.train()


def run_model(args, G_data):
    G_data.split_data()
    net = GNet(G_data.feat_dim, G_data.num_class, args)
    trainer = Trainer(args, net, G_data)
    trainer.train()


def main():
    args = get_args()
    print(args)
    set_random(args.seed)
    start = time.time()
    G_data = FileLoaderNew(args).load_data()
    print('load data using ------>', time.time()-start)

    # if args.fold == 0:
    #     for fold_idx in range(10):
    #         print('start training ------> fold', fold_idx+1)
    #         app_run(args, G_data, fold_idx)
    # else:
    #     print('start training ------> fold', args.fold)
    #     app_run(args, G_data, args.fold-1)
    run_model(args, G_data)


if __name__ == "__main__":
    main()
