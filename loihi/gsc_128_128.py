import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from neurobench.datasets import SpeechCommands
from neurobench.preprocessing import S2SPreProcessor
from neurobench.postprocessing import choose_max_count

from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark

from lava.proc.io.source import RingBuffer as SpikeGenerator
from lava.proc.io.dataloader import SpikeDataloader
from lava.proc.embedded_io.spike import PyToNxAdapter
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.embedded_io.spike import NxToPyAdapter
from lava.proc.io.sink import RingBuffer as Sink

from lava.lib.dl import netx

from lava.utils.loihi2_profiler_api import Loihi2HWProfiler
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg

import time
import random

NUM_TIMESTEPS = 128 # timesteps for S2S GSC sample
NUM_FEATURES = 128 # features of S2S GSC sample
NUM_CLASSES = 35

class S2S_GSC_DataLoader():
    # DataLoader to feed SpikeDataloader with pre-processed GSC samples
    def __init__(self):
        self.dataset = SpeechCommands(path="./data/speech_commands/", subset="testing")
        self.s2s = S2SPreProcessor()
        config_change = {"hop_length": 125, "n_mels": 128} # 128 timesteps, 128 features
        self.s2s.configure(**config_change)

        self.rand_data = np.random.rand(128, 128)

    def __getitem__(self, idx):
        data, label = self.dataset[idx] # data = (timestep, feature)
        data = data.unsqueeze(0) # (1, timestep, feature)
        data, label = self.s2s((data, label)) # (timestep, feature)
        data = data[1:,:] # remove the first timestep of each sample
        data = data.squeeze() # (timestep, feature)
        data = data.transpose(1, 0) # (feature, timestep)
        data = data.detach().numpy()
        return data, label.item()

    def __len__(self):
        return len(self.dataset)

def shuffle(length):
    l = [i for i in range(length)]
    random.shuffle(l)
    return l

if __name__ == "__main__":
    # dataloader
    s2s_gsc = S2S_GSC_DataLoader()

    data_indexes = shuffle(len(s2s_gsc))
    
    # Lava-Loihi units
    run_config = Loihi2HwCfg()
    
    # Processes for input spike communication
    sg = SpikeGenerator(data=s2s_gsc[data_indexes[0]][0])
    label = s2s_gsc[data_indexes[0]][1]
    
    # sg = SpikeDataloader(dataset=s2s_gsc, interval=(NUM_TIMESTEPS))

    # TODO: check out where can reduce num_message_bits
    # py2nx = PyToNxAdapter(shape=(NUM_FEATURES, ), num_message_bits=2)
    py2nx = PyToNxAdapter(shape=(NUM_FEATURES, ))

    # import hdf5 network using netx
    # Cannot use reset_interval because it must be power of two
    # net = netx.hdf5.Network(net_config='../slayer/gsc/kangaroo_gsc_trained_128_128/network.h5', input_message_bits=2, reset_interval=128)
    net = netx.hdf5.Network(net_config='../slayer/gsc/kangaroo_gsc_trained_128_128/network.h5')
    # Processes for output spike collection
    # nx2py = NxToPyAdapter(shape=(NUM_CLASSES, ), num_message_bits=1)
    nx2py = NxToPyAdapter(shape=(NUM_CLASSES, ))
    output_sink = Sink(shape=(NUM_CLASSES, ), buffer=NUM_TIMESTEPS) # TODO: new output process that accumulates spikes

    sg.s_out.connect(py2nx.inp)
    py2nx.out.connect(net.inp)
    net.out.connect(nx2py.inp)
    nx2py.out.connect(output_sink.a_in)
    
    #####################
    
    # TODO: appears that can only configure profiler for one run() call, any more causes runtime error.
    # ----> approach should be to benchmark one sample (100 times) or combine samples such that run covers the entire test set
    # profiler = Loihi2HWProfiler(run_config)
    # profiler.execution_time_probe(num_steps=NUM_TIMESTEPS, t_start=1, t_end=NUM_TIMESTEPS, dt=1, buffer_size=1024)
    # # profiler.energy_probe(num_steps=num_steps) # energy probe seems to be not available for now
    # profiler.activity_probe()
    # profiler.memory_probe()    

    # # dummy run for runtime to start
    # sg.run(condition=RunSteps(num_steps=NUM_TIMESTEPS), run_cfg=run_config)
    # output_sink.data.set(np.zeros_like(output_sink.data.get()))
    
    times = []
    
    correct = 0
    total = 0

    counter = 5
    print("Starting run.")
    for i in tqdm(data_indexes):
        if counter == 0:
            break
        counter -= 1

        if i != data_indexes[0]:
            data, label = s2s_gsc[i]
            sg.data.set(data)
        
        net.run(condition=RunSteps(num_steps=NUM_TIMESTEPS), run_cfg=run_config)            
        
        spikes = output_sink.data.get()
        print(spikes.sum(axis=0))
        output_sink.data.set(np.zeros_like(spikes))

        # Network should reset itself every 128 timesteps
        # Reset all network state back to zero
        # each block is a Dense connected to LIF
        for block in net.layers:
            block.neuron.u.set(np.zeros_like(block.neuron.u.get()))
            block.neuron.v.set(np.zeros_like(block.neuron.v.get()))
        
        # spikes are in shape (NUM_CLASSES, NUM_TIMESTEPS)
        pred = choose_max_count(torch.tensor(spikes).transpose(0,1).unsqueeze(dim=0))

        print("Predicted:", pred, "Label:", label)
        if pred == label:
            correct += 1
        total += 1
    
    
    net.stop()
    
    # # Performance profiling
    # total_time = profiler.execution_time.sum()
    # times.append(total_time)
    
    # TODO: calculate performance metrics

    # Accuracy
    accuracy = correct / total
    print("Accuracy:", accuracy)