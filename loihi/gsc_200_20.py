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

NUM_TIMESTEPS = 201 # timesteps for S2S GSC sample
NUM_FEATURES = 20 # features of S2S GSC sample
NUM_CLASSES = 35
NUM_SAMPLES = 3 # number of samples from test_set to actually test

class S2S_GSC_DataLoader():
    # DataLoader to feed SpikeDataloader with pre-processed GSC samples
    def __init__(self):
        self.dataset = SpeechCommands(path="./data/speech_commands/", subset="testing")
        self.s2s = S2SPreProcessor()

        self.rand_data = np.random.rand(20, 201)

    def __getitem__(self, idx):
        data, label = self.dataset[idx] # data = (timestep, feature)
        data = data.unsqueeze(0) # (1, timestep, feature)
        # TODO: it appears that during run(), we cannot call the torchaudio transform. Can instead implement the math directly here?
        # --->  or instead it looks like we need to do the complex CPU stuff outside of run() function
        data, label = self.s2s((data, label)) # (1, timestep, feature)
        data = data.squeeze() # (timestep, feature)
        data = data.transpose(1, 0) # (feature, timestep)
        data = data.detach().numpy()
        return data, label.item()

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    # # data in repo root dir
    # test_set = SpeechCommands(path="./data/speech_commands/", subset="testing")
    
    # # batch size of 1 for real deployment
    # test_set_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    
    # # preprocessor
    # S2S = S2SPreProcessor()

    # dataloader
    s2s_gsc = S2S_GSC_DataLoader()
    
    # Lava-Loihi units
    run_config = Loihi2HwCfg()
    
    # Processes for input spike communication
    sg = SpikeGenerator(data=s2s_gsc[0][0])
    
    # sg = SpikeDataloader(dataset=s2s_gsc, interval=(NUM_TIMESTEPS))

    py2nx = PyToNxAdapter(shape=(NUM_FEATURES, ))

    # import hdf5 network using netx
    # Cannot use reset_interval because it must be power of two
    net = netx.hdf5.Network(net_config='../slayer/kangaroo_gsc_trained/network.h5', input_message_bits=2)
    breakpoint()
    # ### Phony Network ###
    # dense_input = Dense(weights=np.random.rand(256, 20))
    # lif1 = LIF(shape=(256, ))
    # dense1 = Dense(weights=np.random.rand(256, 256))
    # lif2 = LIF(shape=(256, ))
    # dense2 = Dense(weights=np.random.rand(256, 256))
    # lif3 = LIF(shape=(256, ))
    # dense3 = Dense(weights=np.random.rand(35, 256))
    # lif4 = LIF(shape=(35, ),
    #            du=0,
    #            dv=1,
    #            vth=1,
    #            bias_mant=20,
    #            bias_exp=4)

    # Processes for output spike collection
    nx2py = NxToPyAdapter(shape=(NUM_CLASSES, ))
    output_sink = Sink(shape=(NUM_CLASSES, ), buffer=NUM_TIMESTEPS)
    
    # Connect components
    # sg.s_out.connect(py2nx.inp)
    # py2nx.out.connect(dense_input.s_in)
    # dense_input.a_out.connect(lif1.a_in)
    # lif1.s_out.connect(dense1.s_in)
    # dense1.a_out.connect(lif2.a_in)
    # lif2.s_out.connect(dense2.s_in)
    # dense2.a_out.connect(lif3.a_in)
    # lif3.s_out.connect(dense3.s_in)
    # dense3.a_out.connect(lif4.a_in)
    # lif4.s_out.connect(nx2py.inp)
    # nx2py.out.connect(output_sink.a_in)

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
    
    # # Start runtime, kinda jank
    # # TODO: this run causes the timestep to be offset by 1, which causes source spikes to be misaligned
    # sg.run(condition=RunSteps(num_steps=1), run_cfg=run_config)
    # output_sink.data.set(np.zeros_like(output_sink.data.get()))

    # # Reset all network state back to zero
    # # each block is a Dense connected to LIF
    # for block in net.layers:
    #     block.neuron.u.set(np.zeros_like(block.neuron.u.get()))
    #     block.neuron.v.set(np.zeros_like(block.neuron.v.get()))

    
    times = []
    
    correct = 0
    total = 0
    
    # counter = NUM_SAMPLES
    # # for data in tqdm(test_set_loader):
    # for data in test_set_loader:
    #     if counter == 0:
    #         break
    #     counter -= 1
    
    #     if type(data) is not tuple:
    #         data = tuple(data)
    
    #     data, label = S2S(data) # comes out with shape (batch, timestep, feature)
    #     data = data.squeeze()
    #     data = data.transpose(1, 0) # convert to shape (feature, timestep)
    #     data = data.detach().numpy()
    
    #     assert(data.shape == (NUM_FEATURES, NUM_TIMESTEPS))
    
    #     sg.data.set(data)
    
    #     sg.run(condition=RunSteps(num_steps=NUM_TIMESTEPS), run_cfg=run_config)            
    
    #     spikes = output_sink.data.get()
    #     output_sink.data.set(np.zeros_like(spikes))

    #     # Reset all network state back to zero
    #     # each block is a Dense connected to LIF
    #     for block in net.layers:
    #         block.neuron.u.set(np.zeros_like(block.neuron.u.get()))
    #         block.neuron.v.set(np.zeros_like(block.neuron.v.get()))
        
    #     # spikes are in shape (NUM_CLASSES, NUM_TIMESTEPS)
    #     pred = choose_max_count(torch.tensor(spikes).transpose(0,1).unsqueeze(dim=0))
    #     print("Predicted:", pred.item(), "Label:", label.item())
    #     if pred == label.item():
    #         correct += 1
    #     total += 1

    counter = 5
    print("Starting run.")
    for i in tqdm(range(len(s2s_gsc))):
        if counter == 0:
            break
        counter -= 1

        starttime = time.time()
        net.run(condition=RunSteps(num_steps=NUM_TIMESTEPS), run_cfg=run_config)            
        endtime = time.time()
        
        # # TODO
        # print("Total", endtime-starttime, "Execution", sum(profiler.execution_time))
        # net.stop()
        # quit()
        # #
        
        spikes = output_sink.data.get()
        output_sink.data.set(np.zeros_like(spikes))

        # Reset all network state back to zero
        # each block is a Dense connected to LIF
        for block in net.layers:
            block.neuron.u.set(np.zeros_like(block.neuron.u.get()))
            block.neuron.v.set(np.zeros_like(block.neuron.v.get()))
        
        # spikes are in shape (NUM_CLASSES, NUM_TIMESTEPS)
        pred = choose_max_count(torch.tensor(spikes).transpose(0,1).unsqueeze(dim=0))

        print("Predicted:", pred.item())
        
        # TODO: need to figure out how to extract the label for this sample. 
        #       --> could use an outputprocess but would rather not
        # print("Predicted:", pred.item(), "Label:", label.item())
        # if pred == label.item():
        #     correct += 1
        # total += 1
    
    
    net.stop()
    
    # # Performance profiling
    # total_time = profiler.execution_time.sum()
    # times.append(total_time)
    
    # TODO: calculate performance metrics

    # Accuracy
    accuracy = correct / total
    print("Accuracy:", accuracy)