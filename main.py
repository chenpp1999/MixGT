from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import argparse
import yaml
import torch
import numpy as np
import csv,codecs
from lib.stsutils import get_adjacency_matrix, load_dataset
from model.supervisor import Supervisor
import os
import pandas as pd
TF_ENABLE_ONEDNN_OPTS = 0

parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', default='data/model/PEMS08.yaml', type=str,
                    help='Configuration filename for restoring the model.')
parser.add_argument('--id_filename', default=None, type=str,
                    help='Id filename for dataset')
args = parser.parse_args()


def _init_seed(SEED=10):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main(args):
    with open(args.config_filename) as f:
        config = yaml.safe_load(f)
        adj = get_adjacency_matrix(distance_df_filename=config['data']['sensors_distance'],
                                   num_of_vertices=config['model']['num_nodes'],
                                   type_=config['model']['construct_type'],
                                   id_filename=args.id_filename)
        adj_mx = torch.FloatTensor(adj)
        # a1 = pd.DataFrame(adj_mx.cpu().numpy())
        # a1.to_csv('adj_ori.csv')


        dataloader = load_dataset(dataset_dir=config['data']['data'],
                                  normalizer=config['data']['normalizer'],
                                  batch_size=config['data']['batch_size'],
                                  valid_batch_size=config['data']['batch_size'],
                                  test_batch_size=config['data']['batch_size'],
                                  column_wise=config['data']['column_wise'])
        supervisor = Supervisor(adj_mx=adj_mx, dataloader=dataloader, **config)
        # supervisor.train()
        y_pre, y_true = supervisor._predict(12, wr=True)

        # # 测5min 所有时间 131点
        # plt.plot(range(200, 800), y_true[2, 200:800, 131].cpu().numpy(), 'r-', color='#FF0000',
        #          alpha=0.5, linewidth=1, label='TURE')  # 6
        # plt.plot(range(200, 800), y_pre[2, 200:800, 131].cpu().numpy(), 'r-', color="olive",
        #          alpha=0.5, linewidth=1, label='LSMT')  # 6

        # 所有空间 2142点
        # plt.plot(range(len(y_true[11, 2142, :])), y_true[11, 2142, :].cpu().numpy(), 'r-', color='#FF0000',
        #          alpha=0.5, linewidth=1, label='TURE')  # 6
        # plt.plot(range(len(y_true[11, 2142, :])), y_pre[11, 2142, :].cpu().numpy(), 'r-', color="olive",
        #          alpha=0.5, linewidth=1, label='LSMT')  # 6
        #
        # # 所有时间
        # yt1 = y_true[2, :, 131]
        # yp1 = y_pre[2, :, 131]
        #
        # yt_p1 = []
        # with codecs.open('{}.csv'.format('h3_MG_pems08_results'), 'wb', 'gbk') as f:
        #     writer = csv.writer(f)
        #     for i in range(len(yt1)):
        #         yt_p1.append([yt1[i].cpu().numpy(), yp1[i].cpu().numpy()])
        #         writer.writerow(yt_p1[i])



        # 空间
        # yr1 = y_true[11, 2200:2500, :].cpu().numpy()
        # # yp1 = y_pre[11, 300:600, :].cpu().numpy()
        # yp1 = y_pre[11, 2200:2500, :].cpu().numpy()
        #
        # Rdatas = pd.DataFrame(yr1)
        # Rdatas.to_csv('Pems08_spacetime_R_results.csv')
        # pdatas = pd.DataFrame(yp1)
        # pdatas.to_csv('Pems08_spacetime_P_results.csv')

        # yt_p1 = []
        # with codecs.open('{}.csv'.format('h12_LEISN_pems08_Space'), 'wb', 'gbk') as f:
        #     writer = csv.writer(f)
        #     for i in range(len(yt1)):
        #         yt_p1.append([yt1[i].cpu().numpy(), yp1[i].cpu().numpy()])
        #         writer.writerow(yt_p1[i])

        # 城市节点轴
        # plt.plot(range(0, 170), np.mean(y_true[0:3, 6, ].cpu().numpy(), axis=0), 'r-', color='#FF0000', alpha=0.5, linewidth=1, label='label')
        # plt.plot(range(0, 170), np.mean(y_pre[0:3, 6, ].cpu().numpy(), axis=0), 'r-', color='#4169E1', alpha=1, linewidth=1, label='predict')

        #
        # plt.plot(range(0, 12*36), y_true[0:12, 0:36, 5].reshape(36*12).cpu().numpy(), 'r-', color='#FF0000', alpha=0.5, linewidth=1, label='label')
        # plt.plot(range(0, 12*36), y_pre[0:12, 0:36, 5].reshape(36*12).cpu().numpy(), 'r-', color='#4169E1', alpha=1, linewidth=1, label='predict')
        plt.legend(loc="upper right")
        plt.xlabel('time')
        plt.ylabel('flow')

        plt.title("pems08")  # 标题

        plt.show()

if __name__ == '__main__':
    _init_seed(10)
    main(args)
