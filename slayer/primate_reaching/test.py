import os
import glob
import zipfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, Subset

import lava.lib.dl.slayer as slayer

from dataset import PrimateReaching
from neurobench.models.torch_model import TorchModel

from train import Network

from neurobench.benchmarks import Benchmark

'''
Model was trained / validated with 50 timestep sequences, using linearly weighted MSE loss

Model will be tested on completely sequential timesteps, using R2 score
'''

class Model(TorchModel):
    def __init__(self, net):
        super(Model, self).__init__(net)
        net.eval()

    def __call__(self, x):
        # x is in shape (timesteps, 1, 96)
        # network expects (1, 96, timesteps)
        x = x.permute(1, 2, 0).to(device)

        pred = self.net(x)
        # network returns (1, 2, timesteps), we want (timesteps, 2)
        pred = pred.permute(0, 2, 1)
        pred = pred.squeeze().detach().to('cpu')

        return pred


if __name__ == '__main__':
    # files = ["indy_20160622_01", "indy_20160630_01", "indy_20170131_02",
    #          "loco_20170210_03", "loco_20170215_02", "loco_20170301_05"]
    files = ["indy_20160622_01"]

    for file in files:
        print("Processing", file)

        device = torch.device('cuda')

        if "indy" in file:
            net = Network(input_size=96).to(device)
        else:
            net = Network(input_size=192).to(device)

        net.load_state_dict(torch.load(f'trained_{file}/network.pt'))

        data_dir = './data/primate_reaching/'
        dataset = PrimateReaching(file_path=data_dir, filename=file, num_steps=1, label_series=False,
                                  train_ratio=0.5, bin_width=0.004, biological_delay=0,
                                  remove_segments_inactive=False)

        test_set = Subset(dataset, dataset.ind_test)
        test_loader = DataLoader(dataset=test_set, batch_size=len(dataset), shuffle=False)

        # transpose postprocessor, since lava-dl has timesteps as last dim
        # postprocessor = lambda x: (x[0].transpose(1, 2).to(device), x[1].to(device))

        model = Model(net)

        workload_metrics = ["r2"]

        benchmark = Benchmark(model, test_loader, [], [], [[], workload_metrics])

        results = benchmark.run()
        print(results)
