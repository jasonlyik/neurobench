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

class Network(torch.nn.Module):
    def __init__(self, threshold=1.25, current_decay=0.25, voltage_decay=0.03, tau_grad=0.03, scale_grad=3):
        super(Network, self).__init__()

        neuron_params = {
                'threshold'     : threshold,
                'current_decay' : current_decay,
                'voltage_decay' : voltage_decay,
                'tau_grad'      : tau_grad,
                'scale_grad'    : scale_grad,
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

def main():
    wandb.init(mode="disabled")
    # wandb.init()

    device = torch.device('cuda')

    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs
    lr = wandb.config.lr
    neuron_threshold = wandb.config.neuron_threshold
    current_decay = wandb.config.current_decay
    voltage_decay = wandb.config.voltage_decay
    tau_grad = wandb.config.tau_grad
    scale_grad = wandb.config.scale_grad


    # net = Network().to(device)
    net = Network(
        threshold=neuron_threshold,
        current_decay=current_decay,
        voltage_decay=voltage_decay,
        tau_grad=tau_grad,
        scale_grad=scale_grad
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    data_dir = './data/speech_commands/'
    training_set = SpeechCommands(path=data_dir, subset="training")
    val_set = SpeechCommands(path=data_dir, subset="validation")

    s2s = S2SPreProcessor()

    train_loader = DataLoader(
            dataset=training_set, batch_size=batch_size, shuffle=True
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
        
        if not stats.testing.best_accuracy:
            print("No improvement in accuracy")

        val_acc = stats.testing.accuracy
        wandb.log({
            "epoch": epoch,
            "train_loss": stats.training.loss,
            "train_acc": stats.training.accuracy,
            "val_loss": stats.testing.loss,
            "val_acc": val_acc})

if __name__ == '__main__':
    
    # sweep_configuration = {
    #     "method": "random",
    #     "name": "sweep",
    #     "metric": {"goal": "maximize", "name": "val_acc"},
    #     "parameters": {
    #         "batch_size": {"values": [16, 32, 64]},
    #         "epochs": {"values": [50]},
    #         "lr": {{"values": [.0001, 0.00005, 0.0002]}},
    #         "neuron_threshold": {"values": [1.25, 1.5, 1.0]},
    #         "current_decay": {"values": [0.25, 0.15, 0.35]},
    #         "voltage_decay": {"values": [0.03, 0.05, 0.01]},
    #         "tau_grad": {"values": [0.03, 0.05, 0.01]},
    #         "scale_grad": {"values": [3, 4, 2]},
    #     },
    # }


    # TODO: not looking at batch_size or epochs right now
    # TODO: not taking into account dropout, whether threshold is trainable
    # TODO: also maximizing for val_acc now, and not minimizing spike count
    sweep_configuration = {
        "method": "bayes",
        "name": "disabled",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            "batch_size": {"values": [32, 64]},
            "epochs": {"values": [50]},
            "lr": {"min": 0.0005, "max": 0.002},
            "neuron_threshold": {"min": 1.0, "max": 1.5},
            "current_decay": {"min": 0.15, "max": 0.35},
            "voltage_decay": {"min": 0.01, "max": 0.05},
            "tau_grad": {"min": 0.01, "max": 0.05},
            "scale_grad": {"min": 2, "max": 4},
        },
        "run_cap": 2,
    }

    sweep_id = wandb.sweep(sweep_configuration, project="slayer_gsc")

    wandb.agent(sweep_id, function=main)

    # wandb.init(project='slayer_gsc')

    # trained_folder = 'cluster_gsc_trained'
    # os.makedirs(trained_folder, exist_ok=True)

    # if stats.testing.best_accuracy:
    #     torch.save(net.state_dict(), trained_folder + '/network.pt')
    # stats.update()
    # stats.save(trained_folder + '/')
    # stats.plot(path=trained_folder + '/')
    # net.grad_flow(trained_folder + '/')