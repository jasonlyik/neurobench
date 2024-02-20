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
from neurobench.postprocessing import choose_max_count

from tqdm import tqdm

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
        # # neuron_params_drop = {
        # #         **neuron_params,
        # #         'dropout' : slayer.neuron.Dropout(p=0.05),
        # #     }
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
                    neuron_params_drop, 20, 256,
                ),
                slayer.block.cuba.Dense(
                    neuron_params_drop, 256, 256,
                ),
                slayer.block.cuba.Dense(
                    neuron_params_drop, 256, 256,
                ),
                slayer.block.cuba.Dense(
                    neuron_params, 256, 35,
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
    # device = torch.device('cpu')
    device = torch.device('cuda')

    net = Network().to(device)
    net.load_state_dict(torch.load('kangaroo_gsc_trained/network.pt'))

    data_dir = './data/speech_commands/'
    test_set = SpeechCommands(path=data_dir, subset="testing")

    s2s = S2SPreProcessor()
    test_loader = DataLoader(dataset=test_set, batch_size=256, shuffle=True)

    total = 0
    correct = 0

    for data in tqdm(test_loader):
        spikes, label = s2s(data)
        spikes = spikes.transpose(1, 2) # lava-dl expects timesteps last dim

        spikes = spikes.to(device)

        net.eval()

        output, count = net(spikes)

        pred = choose_max_count(output.transpose(1, 2).to("cpu"))

        total += label.size(0)
        correct += (pred == label).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')
