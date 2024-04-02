import numpy as np
import torch

from lava.proc.io.dataloader import SpikeDataloader
from lava.proc.embedded_io.spike import PyToNxAdapter
from lava.proc.dense.process import Dense
from lava.proc.sdn.process import Sigma
from lava.proc.embedded_io.state import Read as Read_EIO
from interval_sink_new import IntervalReadAccuracy, IntervalReadAccuracyModel

from lava.lib.dl import netx

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg

from lava.utils.loihi2_profiler import Loihi2Activity, Loihi2ExecutionTime, Loihi2Memory, Loihi2Power

import time
import random
from tqdm import tqdm

import os

NUM_TIMESTEPS = 256 # timesteps for S2S GSC sample
NUM_FEATURES = 20 # features of S2S GSC sample
NUM_CLASSES = 35

PREPROCESSED_DATA_PATH = '/homes/jyik/work/s2s_gsc_data'

class S2S_PreProcessed_GSC():
    '''
    Iterable which returns preprocessed GSC samples, which are 256 features, 20 timesteps 
    The preprocessed data has batch size of 500 and is the GSC test set only.
    The data has been shuffled with random seed of 41.
    '''
    def __init__(self, path):
        self.path = path
        self.count = 0

        self.current_file_num = 0
        self.spikes_all = np.load(f'{self.path}/spikes_{self.current_file_num}.npy')
        self.targets_all = np.load(f'{self.path}/targets_{self.current_file_num}.npy')

    def __len__(self):
        return 11005

    def __getitem__(self, idx):
        file_num = int(idx / 500)
        idx = idx % 500

        if file_num != self.current_file_num:
            self.current_file_num = file_num
            self.spikes_all = np.load(f'{self.path}/spikes_{file_num}.npy')
            self.targets_all = np.load(f'{self.path}/targets_{file_num}.npy')

        spikes = self.spikes_all[idx].transpose()
        target = self.targets_all[idx]

        assert(spikes.shape == (NUM_FEATURES, NUM_TIMESTEPS))
        assert(type(target) == np.int64)

        return spikes, target

if __name__ == "__main__": 
    dataset = S2S_PreProcessed_GSC(PREPROCESSED_DATA_PATH)
    dataloader = SpikeDataloader(dataset=dataset, interval=NUM_TIMESTEPS)

    py2nx = PyToNxAdapter(shape=(NUM_FEATURES, ))

    # import hdf5 network using netx
    # TODO: get this model converted and uploaded from kangaroo, see if the overall thing runs right
    net = netx.hdf5.Network(net_config='../slayer/gsc/kangaroo_gsc_trained_256_20/network.h5', reset_interval=NUM_TIMESTEPS)
    
    # by default, reset offset is 1+layer_index, which pipelines the reset. Since we don't want leftover spikes from the previous sample, we
    #    will reset the whole network all at once.
    for block in net.layers:
        block.neuron.proc_params.overwrite("reset_offset", 1)

    # attach network output to Sigma
    readout = Dense(weights=np.eye(NUM_CLASSES))
    counter = Sigma(shape=(NUM_CLASSES,))

    read_eio = Read_EIO(shape=(NUM_CLASSES,), interval=NUM_TIMESTEPS, offset=NUM_TIMESTEPS-1)
    
    read = IntervalReadAccuracy(shape=(NUM_CLASSES,), interval=NUM_TIMESTEPS, offset=NUM_TIMESTEPS-1)

    dataloader.s_out.connect(py2nx.inp)
    py2nx.out.connect(net.inp)
    net.out.connect(readout.s_in)
    readout.a_out.connect(counter.a_in)
    read_eio.connect_var(counter.sigma)
    read_eio.out.connect(read.inp_pred)
    dataloader.ground_truth.connect(read.inp_label)
    
    # default partition allows 2 min
    os.environ['PARTITION'] = "oheogulch_2h"

    samples_to_process = 1000
    timesteps_to_run = samples_to_process * NUM_TIMESTEPS

    # TODO: figure out profilers
    # Note: over 50 samples, no profilers, avg runtime is 3.8744seconds. With three profilers, 40 samples, 
    #       no execution_time_profiler, time is 4.325. TODO: figure out which profilers take time
    # configure profilers
    # activity_profiler = Loihi2Activity()
    # memory_profiler = Loihi2Memory()
    # # TODO: for execution time, increase the bin_size in order to make sure buffer doesn't fill up
    # # execution_time_profiler = Loihi2ExecutionTime(t_start=1, t_end=timesteps_to_run, buffer_size=1024, bin_size=1)
    # power_profiler = Loihi2Power(num_steps=timesteps_to_run)
    # # run_config = Loihi2HwCfg(callback_fxs=[activity_profiler, memory_profiler, execution_time_profiler, power_profiler])
    # run_config = Loihi2HwCfg(callback_fxs=[activity_profiler, memory_profiler, power_profiler])

    run_config = Loihi2HwCfg()

    print("Starting run.")
    start = time.time()
    net.run(condition=RunSteps(num_steps=timesteps_to_run), run_cfg=run_config)
    end = time.time()
    correct = read.correct.get()
    total = read.total.get()
    
    net.stop()
    print("Accuracy:", correct / total)
    print("Time taken:", end - start)
    print("Avg time per sample:", (end - start) / samples_to_process)

    # TODO: pull information from the profilers