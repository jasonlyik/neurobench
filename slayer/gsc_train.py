# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import os
import glob
import zipfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

import lava.lib.dl.slayer as slayer

from neurobench.datasets.speech_commands import SpeechCommands
from neurobench.preprocessing import S2SPreProcessor

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

class Network(torch.nn.Module):
    def __init__(self):
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
                'requires_grad' : True,
            }
        neuron_params_drop = {
                **neuron_params,
                'dropout' : slayer.neuron.Dropout(p=0.05),
            }
        neuron_params_drop = {**neuron_params}

        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Dense(
                    neuron_params_drop, 20, 256, weight_norm=True,
                ),
                slayer.block.cuba.Dense(
                    neuron_params_drop, 256, 256, weight_norm=True,
                ),
                slayer.block.cuba.Dense(
                    neuron_params_drop, 256, 256, weight_norm=True,
                ),
                slayer.block.cuba.Dense(
                    neuron_params, 256, 35, weight_norm=True,
                ),
            ])

    def forward(self, spike):
        count = []
        for block in self.blocks:
            spike = block(spike)
            count.append(torch.mean(spike).item())
        return spike, torch.FloatTensor(count).reshape(
            (1, -1)
        ).to(spike.device)

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
    trained_folder = 'kangaroo_gsc_trained_weight_norm'
    os.makedirs(trained_folder, exist_ok=True)

    # device = torch.device('cpu')
    device = torch.device('cuda')

    net = Network().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    data_dir = './data/speech_commands/'
    training_set = SpeechCommands(path=data_dir, subset="training")
    val_set = SpeechCommands(path=data_dir, subset="validation")

    s2s = S2SPreProcessor()

    train_loader = DataLoader(
            dataset=training_set, batch_size=BATCH_SIZE, shuffle=True
        )
    test_loader = DataLoader(dataset=val_set, batch_size=256, shuffle=True)

    error = slayer.loss.SpikeRate(
            true_rate=0.25, false_rate=0.025, reduction='sum'
        ).to(device)

    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
            net, error, optimizer, stats,
            classifier=slayer.classifier.Rate.predict, count_log=True
        )

    epochs = EPOCHS

    # Just run for more epochs
    epochs = epochs + 100

    wandb.init(project="slayer_gsc")

    for epoch in range(epochs):
        for i, data in enumerate(train_loader):  # training loop
            spikes, label = s2s(data)
            spikes = spikes.transpose(1, 2) # lava-dl expects timesteps last dim

            output, count = assistant.train(spikes, label)
            header = [
                    'Event rate : ' +
                    ', '.join([f'{c.item():.4f}' for c in count.flatten()])
                ]
            stats.print(epoch, iter=i, header=header, dataloader=train_loader)

        for i, data in enumerate(test_loader):  # testing loop
            spikes, label = s2s(data)
            spikes = spikes.transpose(1, 2) # lava-dl expects timesteps last dim

            output, count = assistant.test(spikes, label)
            header = [
                    'Event rate : ' +
                    ', '.join([f'{c.item():.4f}' for c in count.flatten()])
                ]
            stats.print(epoch, iter=i, header=header, dataloader=test_loader)

        wandb.log({
            "epoch": epoch,
            "train_loss": stats.training.loss,
            "train_acc": stats.training.accuracy,
            "val_loss": stats.testing.loss,
            "val_acc": stats.testing.accuracy})

        if stats.testing.best_accuracy:
            torch.save(net.state_dict(), trained_folder + '/network.pt')
        stats.update()
        stats.save(trained_folder + '/')
        stats.plot(path=trained_folder + '/')
        net.grad_flow(trained_folder + '/')