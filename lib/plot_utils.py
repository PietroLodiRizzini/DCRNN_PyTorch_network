import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

plt.rcParams.update({'figure.max_open_warning': 0})

def get_active_nodes_mask(x):
    """
    x: can have shape [batch_dim, num_timesteps_in, num_nodes, num_node_features]
                      or [num_timesteps_in, num_nodes, num_node_features]
    """
    if len(x.shape) == 4:
        return (x[:, :, :, 0] == 1)[0][0]
    elif len(x.shape) == 3:
        return (x[:, :, 0] == 1)[0]
    
def get_active_nodes_ids(x):
    mask = get_active_nodes_mask(x)
    return np.where(mask == 1)[0]

def dragonfly_node_id(graph_node_id):
    """
    Returns the dragonfly node id for the given graph node id
    e.g. In 72 nodes dragonfly, graph node 5 corres–ponds to dragonfly node 0
    Note: graph_node_id must be a graph node id corresponding to a compute node.
    """
    num_ports_per_router = 7
    num_nodes_per_router = 2
    dragonfly_port_1st_computing_node = 5 # the router port to which the first computing node is connected, 5 for 72 nodes dragonfly
    dragonfly_port_id = graph_node_id % num_ports_per_router
    dragonfly_router_id = (graph_node_id - dragonfly_port_id) / num_ports_per_router
    dragonfly_node_id = dragonfly_router_id * num_nodes_per_router + (dragonfly_port_id - dragonfly_port_1st_computing_node)
    return int(dragonfly_node_id)


def process_snapshot(snapshot, prediction, truth, path, active_nodes):
    y_pred_series = prediction[:, snapshot, :]
    y_true_series = truth[:, snapshot, :]

    for node in active_nodes[0:5]:
        fig, ax = plt.subplots()
        ax.plot(y_pred_series[:, node], label='Prediction')
        ax.plot(y_true_series[:, node], label='Ground Truth')
        ax.set_title(f"Graph Node {node} (Dragonfly node {dragonfly_node_id(node)}) Prediction vs. Ground Truth")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.legend()

        plot_dir = os.path.join(path, f"single_outputs/snap_{snapshot}/")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.savefig(os.path.join(plot_dir, f'plot_node_{node}.png'))
        plt.close(fig)


def store_pred_vs_truth_plots(x, y, y_pred, truth, path):
    """
        x: [batch_size, num_timesteps_out, num_nodes, num_node_features]
        y, y_pred: [batch_size, num_timesteps_out, num_nodes, 1]
    """
    active_graph_nodes_ids = get_active_nodes_ids(x)
    
    
    #plot_sliding_window_avg(y_pred, y, epoch_num, path, active_graph_nodes_ids)
    plot_sliding_window(y_pred, truth, path, active_graph_nodes_ids)

    num_timesteps_out = y.shape[1]
    if num_timesteps_out > 1:
        num_active_nodes = len(active_graph_nodes_ids)
        interval = int(num_timesteps_out * .9) if num_timesteps_out != 1 else 1
        #ub = min(y_pred.shape[1], max_snapshots*interval)
        ub = y_pred.shape[1]
        snapshots = range(0, ub, interval)

        for s in snapshots:
            process_snapshot(s, y_pred, truth, path, active_graph_nodes_ids)

        #with Pool(processes=10) as pool: 
        #    pool.starmap(process_snapshot, [(s, y_pred, truth, path, epoch_num, active_graph_nodes_ids) for s in snapshots])

def plot_sliding_window_avg(prediction, y, path, active_nodes):
    # prediction: [num_timesteps_out, num_nodes]
    # truth: [num_timesteps_out, num_nodes]

    for node in active_nodes:
        dragonfly_node_id_val = dragonfly_node_id(node)
        pred = []
        truth_plot = []

        # Define the maximum value of t
        max_t = prediction.shape[1]

        for t in range(max_t):
            indices = []
            if t < prediction.shape[0]:
                min_i = 0
            else:
                min_i = t - prediction.shape[0] + 1
            max_i = t

            for i in range(min_i, max_i + 1):
                j = t - i
                indices.append((i, j))
            
            sum = 0
            for (i, j) in indices:
                sum = sum + prediction[j, i, node]
            mean_prediction = sum / len(indices)

            pred.append(mean_prediction)

        #for i in range(0, truth.shape[1], truth.shape[0]):
        #    truth_plot.extend(truth[:, i, node])
        num_ts_out = y.shape[1]
        for i in range(0, y.shape[0], num_ts_out):
            if i + num_ts_out > y.shape[0]:
                truth_plot.extend(y[i, 0 : y.shape[0] - i, node, 0])
            else:
                truth_plot.extend(y[i, :, node, 0])


        fig, ax = plt.subplots()
        ax.plot(pred, label='Prediction')
        ax.plot(truth_plot, label='Ground Truth')
        ax.set_title(f"Graph Node {node} (Dragonfly node {dragonfly_node_id_val}) Prediction vs. Ground Truth")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.legend()

        plot_dir = os.path.join(path, f"sliding_window_avg")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.savefig(os.path.join(plot_dir, f'node_{node}.png'))
        plt.close(fig)

def plot_sliding_window(y_pred, y, path, active_nodes):
    # y_pred: [num_timesteps_out, num_windows, num_nodes]
    # y:      [num_timesteps_out, num_windows, num_nodes]
    # 50, 285, 252
    num_timesteps_out = y_pred.shape[0]
    num_windows = y_pred.shape[1]

    for node in active_nodes:
        dragonfly_node_id_val = dragonfly_node_id(node)
        plot_y_pred = []
        plot_y_real = []
        # 0, 50, 100, ..., 250
        for i in range(0, num_windows, num_timesteps_out):
            ub_1 = num_timesteps_out
            if i + num_timesteps_out > num_windows: # 300 > 285
                ub_1 = num_windows - i
            
            plot_y_pred.extend(y_pred[:ub_1, i, node])
            plot_y_real.extend(y[:ub_1, i, node])

        fig, ax = plt.subplots()
        
        ax.plot(plot_y_pred, label='Prediction')
        ax.plot(plot_y_real, label='Ground Truth')
        ax.set_title(f"Graph Node {node} (Dragonfly node {dragonfly_node_id_val}) Prediction vs. Ground Truth")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.legend()

        plot_dir = os.path.join(path, f"sliding_window")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.savefig(os.path.join(plot_dir, f'node_{node}.png'))
        plt.close(fig)

def store_loss_plot(train_loss_history, val_loss_history, save_path):
    """
    Stores a png of the plot to save_path
    """
    epochs = range(1, len(train_loss_history) + 1)
    plt.clf()
    
    plt.plot(epochs, train_loss_history, '-', color="orange", label='Training loss')
    plt.plot(epochs, val_loss_history, '-', color="blue", label='Validation loss')
    
    plt.title('Training / validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    save_path = os.path.join(save_path, f"loss")
    plt.savefig(save_path)
    plt.close()