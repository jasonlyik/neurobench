# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import os
import glob
import zipfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

import lava.lib.dl.slayer as slayer

# from neurobench.datasets import PrimateReaching
from dataset import PrimateReaching

import wandb

# Hyperparameters found via wandb sweep
BATCH_SIZE = 32
LEARNING_RATE = 0.00019202967172871096 # TODO: another sweep with higher learning rate?
THRESHOLD = 1.0416058335115037
CURRENT_DECAY = 0.20891491883617025
VOLTAGE_DECAY = 0.04195080192500501
TAU_GRAD = 0.015719635220654594
SCALE_GRAD = 4.0
EPOCHS = 50

# TODO: here, can add an event sparsity loss to the dense layer, but the goal is to maintain velocity
#       so sparsity should be already built in. Can look into this
def event_rate_loss(x, max_rate=0.01):
    mean_event_rate = torch.mean(torch.abs(x))
    return F.mse_loss(F.relu(mean_event_rate - max_rate), torch.zeros_like(mean_event_rate))

class weighted_mse_loss():
    def __init__(self):
        pass

    def __call__(self, input, target):
        # linearly weighted MSE loss
        target = target.transpose(1, 2) # lava-dl uses timesteps as last dimension
        assert(input.shape == target.shape)
        num_timesteps = input.shape[-1]
        weights = torch.linspace(0, 1, num_timesteps, device=input.device)
        return torch.mean(weights * ((input - target) ** 2))

def print_stats(epoch, epochs, train_loss, best_train_loss, val_loss, best_val_loss):
    if train_loss == None:
        train_loss = 0.0
    if best_train_loss == None:
        best_train_loss = 0.0
    if val_loss == None:
        val_loss = 0.0
    if best_val_loss == None:
        best_val_loss = 0.0
    print(f'\r[Epoch {epoch:3d}/{epochs}] Train loss: {train_loss:.4f} (best: {best_train_loss:.4f}), Val loss: {val_loss:.4f} (best: {best_val_loss:.4f})', end='')

class Network(torch.nn.Module):
    def __init__(self, input_size=92):
        '''
        input_size: number of input neurons
        persist: whether to use persistent state. False for training, True for testing
        '''
        super(Network, self).__init__()

        # neuron_params = {
        #         'threshold'     : 1.25,
        #         'current_decay' : 0.25,
        #         'voltage_decay' : 0.03,
        #         'tau_grad'      : 0.03,
        #         'scale_grad'    : 3,
        #         'requires_grad' : False,
        #     }
        neuron_params = {
                'threshold'     : THRESHOLD,
                'current_decay' : CURRENT_DECAY,
                'voltage_decay' : VOLTAGE_DECAY,
                'tau_grad'      : TAU_GRAD,
                'scale_grad'    : SCALE_GRAD,
                'requires_grad' : False,
            }
        neuron_params_drop = {
                **neuron_params,
                'dropout' : slayer.neuron.Dropout(p=0.05),
            }
        neuron_params_drop = {**neuron_params}

        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Dense(
                    neuron_params_drop, input_size, 50, weight_norm=False,
                ),
                slayer.block.cuba.Affine(
                    neuron_params_drop, 50, 2, weight_norm=False, dynamics=True,
                ),
            ])

    def forward(self, spike):
        # TODO: here checkout the loss function. if not adding then this probably is much simpler
        for block in self.blocks:
            spike = block(spike)
        return spike

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [
            b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')
        ]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))


if __name__ == '__main__':
    # files = ["indy_20160622_01", "indy_20160630_01", "indy_20170131_02",
    #          "loco_20170210_03", "loco_20170215_02", "loco_20170301_05"]
    files = ["indy_20160622_01"]

    for file in files:
        print("Processing", file)

        trained_folder = 'trained_' + file
        os.makedirs(trained_folder, exist_ok=True)

        # device = torch.device('cpu')
        device = torch.device('cuda')

        if "indy" in file:
            net = Network(input_size=96).to(device)
        else:
            net = Network(input_size=192).to(device)

        data_dir = './data/primate_reaching/'
        dataset = PrimateReaching(file_path=data_dir, filename=file, num_steps=50, label_series=True,
                                  train_ratio=0.5, bin_width=0.004, biological_delay=0,
                                  remove_segments_inactive=False)

        train_set = Subset(dataset, dataset.ind_train) # first 50% of reaches are train
        val_set = Subset(dataset, dataset.ind_val) # next 25% of reaches are val

        train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

        stats = slayer.utils.LearningStats()
        error = weighted_mse_loss()
        optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE) # TODO: PilotNet uses RAdam, check this out

        assistant = slayer.utils.Assistant(
                net, error, optimizer, stats,
            )

        epochs = EPOCHS
        steps = [60, 120, 160] # TODO: PilotNet uses these epoch milestones to reduce learning rate by 10/3

        wandb.init(project="slayer_primate_reaching", mode="offline") # wandb disabled for now

        for epoch in range(epochs):
            # TODO: lr reduction?
            # if epoch in steps:
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] /= 10/3

            for i, data in enumerate(train_loader):  # training loop
                spikes, labels = data
                spikes = spikes.to(device)
                labels = labels.to(device)
                spikes = spikes.transpose(1, 2) # lava-dl expects timesteps last dim

                assistant.train(spikes, labels)
                print_stats(epoch, epochs, stats.training.loss, stats.training.min_loss, stats.testing.loss, stats.testing.min_loss)

            for i, data in enumerate(val_loader):  # testing loop
                spikes, labels = data
                spikes = spikes.to(device)
                labels = labels.to(device)
                spikes = spikes.transpose(1, 2) # lava-dl expects timesteps last dim

                assistant.test(spikes, labels)
                print_stats(epoch, epochs, stats.training.loss, stats.training.min_loss, stats.testing.loss, stats.testing.min_loss)

            wandb.log({
            "epoch": epoch,
            "train_loss": stats.training.loss,
            "val_loss": stats.testing.loss})

            if stats.testing.best_loss:
                torch.save(net.state_dict(), trained_folder + '/network.pt')
            stats.update()
            stats.save(trained_folder + '/')
            net.grad_flow(trained_folder + '/')

            # TODO: checkpoint saves
            # if epoch%10 == 0:
            #     torch.save({'net': net.state_dict(), 'optimizer': optimizer.state_dict()}, trained_folder + f'/checkpoint{epoch}.pt')

