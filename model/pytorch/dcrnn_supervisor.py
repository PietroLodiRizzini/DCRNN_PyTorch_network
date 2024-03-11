import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib import utils
from lib import plot_utils
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.loss import *
import torch.nn as nn

from gpu import gpu

gpu_id = gpu.get_gpu_id()
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
#device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        dataset_dir = self._data_kwargs.get('dataset_dir')
        for replace in ['horizon', 'seq_len', 'input_dim', 'output_dim']:
            value = self._model_kwargs.get(replace)
            dataset_dir = dataset_dir.replace(f"[{replace}]", str(value))
        self._data_kwargs['dataset_dir'] = dataset_dir

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        self._writer = SummaryWriter('runs/' + self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self._logger.info("Start logging")

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        self._active_nodes = utils.load_active_nodes(self._data_kwargs.get('dataset_dir'))

        self.standard_scaler = self._data['scaler']

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder
        self.filter_test_loss = self._model_kwargs.get('filter_test_loss', False)

        # setup model
        dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
        #dcrnn_model = nn.DataParallel(dcrnn_model)
        dcrnn_model = dcrnn_model.cuda(gpu_id) if torch.cuda.is_available() else dcrnn_model
        #dcrnn_model.to(device)

        self.dcrnn_model = dcrnn_model
        
        self._logger.info("Model created")
        self._logger.info(f"Using {device.type}")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

        self.loss = self._model_kwargs.get('loss', 'mae')

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            seq_len = kwargs['model'].get('seq_len')
            num_features = kwargs['model'].get('input_dim')
            num_targets = kwargs['model'].get('output_dim')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            '''run_id = 'dcrnn_%s_%d_h_%d_ %s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))'''
            run_id = f"{filter_type_abbr}_{max_diffusion_step}diffSteps_{num_rnn_layers}rnnLayers_{rnn_units}rnnUnits_{learning_rate}lr_{batch_size}batchSize_{time.strftime('%m%d%H%M%S')}"
            base_dir = kwargs.get('base_dir')
            base_dir = os.path.join(base_dir, f"{seq_len}in_{horizon}out_{num_features}features_{num_targets}targets")
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.dcrnn_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models/epo%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self):
        self._setup_graph()
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
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
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []

            y_truths = []
            y_preds = []

            # mask of len num_nodes (252) that is 1 for node ids that are active, 0 otherwise
            node_mask = self._active_nodes
            num_nodes = len(node_mask)
            num_active_nodes = node_mask.sum()

            for _, (_x, _y) in enumerate(val_iterator):
                x, y = self._prepare_data(_x, _y)

                # x: ([num_timesteps_in, batch_size, num_nodes*num_node_features])
                # y: ([num_timesteps_out, batch_size, num_nodes*num_targets])

                # [num_timesteps_out, batch_size, num_nodes*num_targets]
                output = self.dcrnn_model(x) # [num_timesteps_out, batch_size, num_nodes]

                # filter the output in order to compute the loss only on the desired/active nodes
                if self.filter_test_loss: #and dataset == 'test':
                    output = output[:, :, node_mask == 1] # 50, 64, 36
                    y = y[:, :, node_mask == 1]

                loss = self._compute_loss(y, output)
                losses.append(loss.item())

                num_timesteps_out = y.shape[0]
                batch_size = y.shape[1]
                num_targets = _y.shape[3]

                #y_truths.append(np.reshape(y.cpu(), (num_timesteps_out, batch_size, num_nodes, num_targets))[:, :, :, 0])
                y_truths.append(np.reshape(y.cpu(), (num_timesteps_out, batch_size, num_active_nodes, num_targets))[:, :, :, 0])

                y_preds.append(np.reshape(output.cpu(), (num_timesteps_out, batch_size, num_active_nodes, num_targets))[:, :, :, 0])
                #y_preds.append(output[:, :, :num_nodes].cpu())

            mean_loss = np.mean(losses)

            #self._logger.info(f"shape: {output.shape}")
            #self._logger.info(f"len(y_truths):{len(y_truths)} len(y_preds): {len(y_preds)}")

            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])

                y_truth = y_truths[t]
                y_pred = y_preds[t]

                # filter y_truth and y_pred

                # list of containing batch_num tensors of shape [num_timesteps_out, batch_size, num_nodes] 
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)

            return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}
        
    def evaluate_multiple_metrics(self, dataset='val', batches_seen=0, predict_n_timesteps=1):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            dataset_iterator = self._data['{}_loader'.format(dataset)].get_iterator()

            y_truths = []
            y_preds = []
            y_truths_filtered = [] # contains values only of active nodes
            y_preds_filtered = []  # contains values only of active nodes

            # mask of len num_nodes (252) that is 1 for node ids that are active, 0 otherwise
            node_mask = self._active_nodes
            num_nodes = len(node_mask)
            num_active_nodes = node_mask.sum()

            for _, (_x, _y) in enumerate(dataset_iterator):
                
                # _x: (batch_size, num_timesteps_in, num_nodes, num_node_features)

                x, y = self._prepare_data(_x, _y)

                # x: ([num_timesteps_in, batch_size, num_nodes*num_node_features])
                # y: ([num_timesteps_out, batch_size, num_nodes*num_targets])

                # output: [num_timesteps_out, batch_size, num_nodes*num_targets]
                output = self.dcrnn_model(x)

                num_timesteps_out = output.shape[0]
                batch_size = output.shape[1]
                num_targets = _y.shape[3]

                _output = np.reshape(output.cpu(), (num_timesteps_out, batch_size, num_nodes, num_targets))
                _output = np.transpose(_output, (1, 0, 2, 3)).cpu()
                # iteration-duration is the 0-th element in dim 3 of _output: remove it
                # next_input = np.concatenate([_x[:, 1:, :, :], np.array(_output[:, :, :, 1:])], axis=1)

                # filter the output in order to compute the loss only on the desired/active nodes
                if self.filter_test_loss and dataset == 'test':
                    output_filtered = np.reshape(output.cpu(), (num_timesteps_out, batch_size, num_nodes, num_targets))
                    output_filtered = output_filtered[:, :, node_mask == 1, 0]
                    output_filtered = np.reshape(output_filtered, (num_timesteps_out, batch_size, num_active_nodes))
                    #output_filtered = output[:, :, node_mask == 1] # 50, 64, 36
                    y_filtered = np.reshape(y.cpu(), (num_timesteps_out, batch_size, num_nodes, num_targets))
                    y_filtered = y_filtered[:, :, node_mask == 1, 0]
                    y_filtered = np.reshape(y_filtered, (num_timesteps_out, batch_size, num_active_nodes))
                    #y_filtered = y[:, :, node_mask == 1]
                    # output_filtered, y_filtered: [num_timesteps_out, batch_size, num_active_nodes]
                    y_preds_filtered.append(output_filtered.cpu())
                    y_truths_filtered.append(y_filtered.cpu())
                
                y_truths.append(np.reshape(y.cpu(), (num_timesteps_out, batch_size, num_nodes, num_targets))[:, :, :, 0])
                y_preds.append(np.reshape(output.cpu(), (num_timesteps_out, batch_size, num_nodes, num_targets))[:, :, :, 0])
                #y_truths_filtered.append(y.cpu())
                #y_preds_filtered.append(output.cpu())

            #self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension
            y_preds_filtered = np.concatenate(y_preds_filtered, axis=1)
            y_truths_filtered = np.concatenate(y_truths_filtered, axis=1)  # concatenate on batch dimension

            losses = {loss: self._compute_loss(torch.tensor(y_truths),
                                               torch.tensor(y_preds), loss) for loss in ['mae', 'rmse', 'mape']}

            y_truths_scaled = []
            y_preds_scaled = []
            y_truths_filtered_scaled = []
            y_preds_filtered_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truth_filtered = self.standard_scaler.inverse_transform(y_truths_filtered[t])
                y_pred_filtered = self.standard_scaler.inverse_transform(y_preds_filtered[t])

                y_truth = y_truths[t]
                y_pred = y_preds[t]
                y_truth_filtered = y_truths_filtered[t]
                y_pred_filtered = y_preds_filtered[t]

                # list of containing batch_num tensors of shape [num_timesteps_out, batch_size, num_nodes] 
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)
                y_truths_filtered_scaled.append(y_truth_filtered)
                y_preds_filtered_scaled.append(y_pred_filtered)

            return losses, {'prediction': y_preds_scaled, 'truth': y_truths_scaled, 'prediction_filtered': y_preds_filtered_scaled, 'truth_filtered': y_truths_filtered_scaled}

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=True,
               test_every_n_epochs=10, store_val_plot=True, epsilon=1e-8, loss_plot_every_n_epochs=1, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        #wait_plot = 0
        #wait_plot_thresh = 10
        optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
        #                                                    gamma=lr_decay_ratio)

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        train_loss_history = []
        val_loss_history = []
        #'''
        for epoch_num in range(self._epoch_num, epochs):
            #wait_plot += 1
            self.dcrnn_model = self.dcrnn_model.train()

            train_iterator = self._data['train_loader'].get_iterator()
            losses = []

            start_time = time.time()

            for _, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()

                x, y = self._prepare_data(x, y)

                output = self.dcrnn_model(x, y, batches_seen)

                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

                loss = self._compute_loss(y, output)

                self._logger.debug(loss.item())

                losses.append(loss.item())

                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                #torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)

                optimizer.step()
            self._logger.info("epoch complete")
            #lr_scheduler.step()
            self._logger.info("evaluating now!")
            
            '''    
        for epoch_num in range(self._epoch_num, epochs):
            #wait_plot += 1
            self.dcrnn_model = self.dcrnn_model.train()
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []

            start_time = time.time()

            for _, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()

                x, y = self._prepare_data(x, y)

                # Compute output
                output = self.dcrnn_model(x, y, batches_seen)

                if batches_seen == 0:
                    # This is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

                # Compute loss
                loss = self._compute_loss(y, output)

                # Apply weights
                batch_weights = self._data['train_loader'].weights[_ * self._data['train_loader'].batch_size:(_ + 1) * self._data['train_loader'].batch_size]

                # Multiply loss by batch_weights
                weighted_loss = torch.mean(loss.cpu() * torch.Tensor(batch_weights))

                self._logger.debug(weighted_loss.item())

                losses.append(weighted_loss.item())

                batches_seen += 1
                weighted_loss.backward()

                # Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)

                optimizer.step()

            epoch_loss = np.mean(losses)
            end_time = time.time()

            self._logger.info("Epoch {} completed. Loss: {:.4f}".format(epoch_num, epoch_loss))
            lr_scheduler.step()
            self._logger.info("Evaluating now!")
            #'''

            val_loss, y_val = self.evaluate(dataset='val', batches_seen=batches_seen)
            val_loss_decreased = val_loss < min_val_loss

            if store_val_plot and val_loss_decreased: #and wait_plot > wait_plot_thresh:
                #wait_plot = 0
                prediction = torch.tensor(y_val['prediction'])
                truth = torch.tensor(y_val['truth'])
                
                path = os.path.join(self._log_dir, 'plots/val/')
                plot_utils.store_pred_vs_truth_plots(self._data['x_val'], self._data['y_val'], prediction, truth, self._active_nodes, path)

                #### TRAIN plot
                #train_loss, y_train = self.evaluate(dataset='train', batches_seen=batches_seen)

                #prediction = torch.tensor(y_train['prediction'])
                #truth = torch.tensor(y_train['truth'])
                
                #path = os.path.join(self._log_dir, 'plots/train/')
                #plot_utils.store_pred_vs_truth_plots(self._data['x_train'], self._data['y_train'], prediction, truth, self._active_nodes, path)
                ###

            end_time = time.time()

            train_loss_history.append(np.mean(losses))
            val_loss_history.append(val_loss)

            self._writer.add_scalar('training loss',
                                    np.mean(losses),
                                    batches_seen)

            if (epoch_num % log_every) == log_every - 1:
                '''message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))'''
                
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss, optimizer.param_groups[0]['lr'],
                                           (end_time - start_time))
                
                self._logger.info(message)

            if ((epoch_num % test_every_n_epochs) == test_every_n_epochs - 1 or val_loss_decreased): #and wait_plot > wait_plot_thresh:
                #wait_plot = 0
                # test
                # evaluate model on test data
                test_losses, y = self.evaluate_multiple_metrics(dataset='test', batches_seen=batches_seen)

                # store prediction vs truth plots
                prediction = torch.tensor(y['prediction'])
                truth = torch.tensor(y['truth'])
                path = os.path.join(self._log_dir, 'plots/test/')
                plot_utils.store_pred_vs_truth_plots(self._data['x_test'], self._data['y_test'], prediction, truth, self._active_nodes, path)
                

                # store prediction vs truth values

                '''message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f}, test_rmse: {:.4f}, test_mape: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), test_losses['mae'], test_losses['rmse'], test_losses['mape'], lr_scheduler.get_lr()[0],
                                           (end_time - start_time))'''
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f}, test_rmse: {:.4f}, test_mape: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), test_losses['mae'], test_losses['rmse'], test_losses['mape'],  optimizer.param_groups[0]['lr'],
                                           (end_time - start_time))
                self._logger.info(message)

                
            # store loss plot
            if (epoch_num % loss_plot_every_n_epochs) == loss_plot_every_n_epochs - 1:
                self._logger.info(f"Storing loss plot")
                plot_utils.store_loss_plot(train_loss_history, val_loss_history, self._log_dir)
                
            
            if val_loss_decreased:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

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

    def _compute_loss(self, y_true, y_predicted, loss = None):
        #y_true = self.standard_scaler.inverse_transform(y_true)
        #y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        if loss is None:
            loss = self.loss
        
        if loss == 'mae':
            return masked_mae_loss(y_predicted, y_true)
        elif loss == 'mse':
            return masked_mse_loss(y_predicted, y_true)
        elif loss == 'rmse':
            return masked_rmse_loss(y_predicted, y_true)
        elif loss == 'mape':
            return masked_mape_loss(y_predicted, y_true)
        
