import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib import utils


from model.inceptionModel import InceptionTransformer
from model.loss import masked_mape, masked_rmse, masked_mae, masked_mse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import datasets,transforms


class Supervisor:
    def __init__(self, adj_mx, dataloader, **kwargs):

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._pre_model_kwargs = kwargs.get('pre_model')
        self._train_kwargs = kwargs.get('train')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        self._log_dir = self._get_log_dir(kwargs)

        self._writer = SummaryWriter('runs/' + self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)


        self._data = dataloader
        self.standard_scaler = self._data['scaler']

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 170))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len', 12))
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 12))
        self.l2lambda = float(self._model_kwargs.get('l2lambda', 0))

        self.preModels = ''
        model = InceptionTransformer(adj_mx, self._model_kwargs, self._pre_model_kwargs)
        # model = GRU(170, 12)
        # model = InceptionTransformer(adj_mx, self._logger, self._model_kwargs, self._pre_model_kwargs)
        self.model = model.cuda() if torch.cuda.is_available() else model
        self._logger.info("Model created")
        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('_max_diffusion_step')
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'model_%s_%d_h_%d_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.model.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models/epo%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self):
        self._setup_graph()
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.model = self.model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.model = self.model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            y_truths = []
            y_preds = []
            y_list = []
            output_list = []
            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                batch_size = x.size(1)
                output = self.model(x)
                x1 = self.standard_scaler.inverse_transform(x)
                x_truth = x1.view(self.seq_len, batch_size, self.num_nodes, self.input_dim)[..., :1]
                output_flow = (1 + output) * x_truth[-1].squeeze(-1)
                y_list.append(y)
                output_list.append(output_flow)
                y_truths.append(y.cpu())
                y_preds.append(output.cpu())
            y_pred_full = torch.cat(output_list, dim=1)
            y_true_full = torch.cat(y_list, dim=1)
            mean_loss = self._compute_maeloss(y_true_full, y_pred_full).item()

            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)

            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)
            return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}

    def evaluateTest(self, dataset='test'):
        with torch.no_grad():
            self.model = self.model.eval()
            test_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            y_list = []
            output_list = []
            for _, (x, y) in enumerate(test_iterator):
                x, y = self._prepare_data(x, y)
                batch_size = x.size(1)
                output = self.model(x)
                x1 = self.standard_scaler.inverse_transform(x)
                x_truth = x1.view(self.seq_len, batch_size, self.num_nodes, self.input_dim)[..., :1]
                output_flow = (1 + output) * x_truth[-1].squeeze(-1)
                y_list.append(y)
                output_list.append(output_flow)
            y_pred_full = torch.cat(output_list, dim=1)
            y_true_full = torch.cat(y_list, dim=1)
            # import matplotlib.pyplot as plt
            # plt.plot(range(0, len(y_pred_full[11, 0:600, 11])), y_pred_full[11, 0:600, 11].cpu().numpy(), 'blue', label='pre')
            # plt.plot(range(0, len(y_true_full[11, 0:600, 11])), y_true_full[11, 0:600, 11].cpu().numpy(), 'red', label='real')
            # plt.show()
            # plt.plot(range(0, len(y_pred_full[2, 0:600, 11])), y_pred_full[2, 0:600, 11].cpu().numpy(), 'blue', label='pre')
            # plt.plot(range(0, len(y_true_full[2, 0:600, 11])), y_true_full[2, 0:600, 11].cpu().numpy(), 'red', label='real')
            # plt.show()
            mean_loss = self._compute_maeloss(y_true_full, y_pred_full).item()
            mean_mape = self._compute_mapeloss(y_true_full, y_pred_full).item()
            mean_rmse = self._compute_rmseloss(y_true_full, y_pred_full).item()
            return mean_loss, mean_mape, mean_rmse

    def evaluateTest2(self, dataset='test', num=None, wr=True):
        with torch.no_grad():
            self.gmsdr_model = self.model.eval()
            test_iterator = self._data['{}_loader'.format(dataset)].get_iterator()

            start_time = time.time()
            tqdm_loader = tqdm(enumerate(test_iterator))
            y_list = []
            output_list = []
            for _, (x, y) in tqdm_loader:
                x, y = self._prepare_data(x, y)
                batch_size = x.size(1)
                output = self.gmsdr_model(x)
                x1 = self.standard_scaler.inverse_transform(x)
                x_truth = x1.view(self.seq_len, batch_size, self.num_nodes, self.input_dim)[..., :1]
                output_flow = (1 + output) * x_truth[-1].squeeze(-1)
                y_list.append(y)
                output_list.append(output_flow)
                # print(y.shape)

            y_pred_full = torch.cat(output_list, dim=1)
            # print(y_pred_full.shape)
            y_true_full = torch.cat(y_list, dim=1)


            if num == None:
                mean_loss = self._compute_maeloss(y_true_full, y_pred_full).item()
                mean_mape = self._compute_mapeloss(y_true_full, y_pred_full).item()
                mean_rmse = self._compute_rmseloss(y_true_full, y_pred_full).item()
            else:
                tmep_data = [['horizon', 'mae', 'mape', 'rmse']]
                for i in range(num):
                    mean_loss = self._compute_maeloss(y_true_full[0:i+1,], y_pred_full[0:i+1,]).item()
                    mean_mape = self._compute_mapeloss(y_true_full[0:i+1,], y_pred_full[0:i+1,]).item()
                    mean_rmse = self._compute_rmseloss(y_true_full[0:i+1,], y_pred_full[0:i+1,]).item()
                    tmep_data.append([i+1, mean_loss, mean_mape, mean_rmse])
                    import csv, codecs
                    if wr:
                        with codecs.open('{}.csv'.format('pems08_results'), 'wb', 'gbk') as f:
                            writer = csv.writer(f)
                            for data in tmep_data:
                                writer.writerow(data)
            for i in range(3000):
                # 所有空间
                # mean_loss = self._compute_maeloss(y_true_full[:,:, i ], y_pred_full[:,:,i]).item()
                # mean_mape = self._compute_mapeloss(y_true_full[:,:, i ], y_pred_full[:,:,i]).item()
                # mean_rmse = self._compute_rmseloss(y_true_full[:,:, i ], y_pred_full[:,:,i]).item()
                # 所有时间
                mean_loss = self._compute_maeloss(y_true_full[:, i, :], y_pred_full[:, i, :]).item()
                mean_mape = self._compute_mapeloss(y_true_full[:, i, :], y_pred_full[:, i, :]).item()
                mean_rmse = self._compute_rmseloss(y_true_full[:, i, :], y_pred_full[:, i, :]).item()
                # print("node_num:{}, mae:{}, mape:{}".format(i, mean_loss,mean_mape))
                print("horizon:{},指标：mae {},mape {},rmse {}".format(i, mean_loss, mean_mape, mean_rmse))
            return mean_loss, mean_mape, mean_rmse, y_pred_full, y_true_full

    def _predict(self, numb,patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, wr=True, **kwargs,):
        mae, mape, rmse, y_pre, y_true = self.evaluateTest2(dataset='test', num=numb, wr=wr)
        message = 'Best Val Test: Epoch [{}/{}] mae: {:.4f}, mape: {:.4f},  rmse: {:.4f}, ' \
                  .format(322, epochs, mae, mape, rmse,)

        self._logger.info(message)
        return y_pre, y_true

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):

        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio)

        self._logger.info('Start training ...')
        self._logger.info('pre_k = ' + str(self._model_kwargs.get('pre_k')))
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        for epoch_num in range(self._epoch_num, epochs):

            self.model = self.model.train()

            train_iterator = self._data['train_loader'].get_iterator()
            losses = []

            start_time = time.time()

            tqdm_loader = tqdm(enumerate(train_iterator))
            for _, (x, y) in tqdm_loader:
                optimizer.zero_grad()

                x, y = self._prepare_data(x, y)

                output = self.model(x)

                if batches_seen == 0:
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)

                loss = self._compute_maeloss(y, output)
                lossl2 = self.model.Loss_l2() * self.l2lambda
                loss += lossl2
                self._logger.debug(loss.item())
                losses.append(loss.item())
                batches_seen += 1
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                tqdm_loader.set_description(
                    f'train epoch: {epoch_num:3}/{epochs}, '
                    f'loss: {loss.item():3.6}')

            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")

            val_loss, _ = self.evaluate(dataset='val', batches_seen=batches_seen)

            end_time = time.time()

            self._writer.add_scalar('training loss',
                                    np.mean(losses),
                                    batches_seen)

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.7f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                mae, mape, rmse = self.evaluateTest(dataset='test')
                message = 'EVERY_N_EPOCH Test: Epoch [{}/{}] mae: {:.4f}, rmse: {:.4f},  mape: {:.4f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs,
                                           mae, rmse, mape, (end_time - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                mae, mape, rmse = self.evaluateTest(dataset='test')
                message = 'Best Val Test: Epoch [{}/{}] mae: {:.4f}, mape: {:.4f},  rmse: {:.4f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs,
                                           mae, mape, rmse, (end_time - start_time))

                self._logger.info(message)
            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break
        lr_scheduler.step()

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)

        return x, y

    # def _get_x_y(self, x, y):
    #     """
    #     :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    #     :param y: shape (batch_size, horizon, num_sensor, input_dim)
    #     :returns x shape (seq_len, batch_size, num_sensor, input_dim)
    #              y shape (horizon, batch_size, num_sensor, input_dim)
    #     """
    #     x = torch.from_numpy(x).float()
    #     y = torch.from_numpy(y).float()
    #     self._logger.debug("X: {}".format(x.size()))
    #     self._logger.debug("y: {}".format(y.size()))
    #     x = x.permute(1, 0, 2, 3)
    #     y = y.permute(1, 0, 2, 3)
    #     return x, y

    # def _get_x_y_in_correct_dims(self, x, y):
    #     """
    #     :param x: shape (seq_len, batch_size, num_sensor, input_dim)
    #     :param y: shape (horizon, batch_size, num_sensor, input_dim)
    #     :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
    #              y: shape (horizon, batch_size, num_sensor * output_dim)
    #     """
    #     batch_size = x.size(0)
    #     x = x.view(batch_size, self.seq_len, self.num_nodes * self.input_dim)
    #     x = x.view(batch_size, self.seq_len, self.num_nodes * self.input_dim)
    #     y = y[..., :self.output_dim].view(batch_size, self.horizon,
    #                                       self.num_nodes * self.output_dim)
    #     return x, y

    def _compute_maeloss(self, y_true, y_predicted):
        return masked_mae(y_predicted, y_true, 0.0)

    def _compute_rmseloss(self, y_true, y_predicted):
        return masked_rmse(y_predicted, y_true, 0.0)

    def _compute_mseloss(self, y_true, y_predicted):
        return masked_mse(y_predicted, y_true, 0.0)

    def _compute_mapeloss(self, y_true, y_predicted):
        return masked_mape(y_predicted, y_true, 0.0)
