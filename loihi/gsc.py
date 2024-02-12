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
from lava.proc.embedded_io.spike import PyToNxAdapter
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.embedded_io.spike import NxToPyAdapter
from lava.proc.io.sink import RingBuffer as Sink

from lava.utils.loihi2_profiler_api import Loihi2HWProfiler
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg

NUM_TIMESTEPS = 201 # timesteps for S2S GSC sample
NUM_FEATURES = 20 # features of S2S GSC sample
NUM_CLASSES = 35
NUM_SAMPLES = 5 # number of samples from test_set to actually test

if __name__ == "__main__":
    # data in repo root dir
    test_set = SpeechCommands(path="./data/speech_commands/", subset="testing")
    
    # batch size of 1 for real deployment
    test_set_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    
    # preprocessor
    S2S = S2SPreProcessor()
    
    # Lava-Loihi units
    run_config = Loihi2HwCfg()
    
    # Processes for input spike communication
    sg = SpikeGenerator(data=np.random.rand(20, 201)) # TODO: check number of bits used to store the spikes, should be like 2
    py2nx = PyToNxAdapter(shape=(NUM_FEATURES, ))
    
    ### Phony Network ###
    dense_input = Dense(weights=np.random.rand(256, 20))
    lif1 = LIF(shape=(256, ))
    dense1 = Dense(weights=np.random.rand(256, 256))
    lif2 = LIF(shape=(256, ))
    dense2 = Dense(weights=np.random.rand(256, 256))
    lif3 = LIF(shape=(256, ))
    dense3 = Dense(weights=np.random.rand(35, 256))
    lif4 = LIF(shape=(35, ),
               du=0,
               dv=1,
               vth=1,
               bias_mant=20,
               bias_exp=4)

    # Processes for output spike collection
    # TODO: must batch inputs/outputs to limit IO cost
    nx2py = NxToPyAdapter(shape=(NUM_CLASSES, ))
    output_sink = Sink(shape=(NUM_CLASSES, ), buffer=NUM_TIMESTEPS)
    
    # Connect components
    sg.s_out.connect(py2nx.inp)
    py2nx.out.connect(dense_input.s_in)
    dense_input.a_out.connect(lif1.a_in)
    lif1.s_out.connect(dense1.s_in)
    dense1.a_out.connect(lif2.a_in)
    lif2.s_out.connect(dense2.s_in)
    dense2.a_out.connect(lif3.a_in)
    lif3.s_out.connect(dense3.s_in)
    dense3.a_out.connect(lif4.a_in)
    lif4.s_out.connect(nx2py.inp)
    nx2py.out.connect(output_sink.a_in)
    #####################
    
    # Note: appears that can only configure profiler for one run() call, any more causes runtime error.
    # ----> approach should be to benchmark one sample (100 times) or combine samples such that run covers the entire test set
    # profiler = Loihi2HWProfiler(run_config)
    # profiler.execution_time_probe(num_steps=NUM_TIMESTEPS, t_start=1, t_end=NUM_TIMESTEPS, dt=1, buffer_size=1024)
    # # profiler.energy_probe(num_steps=num_steps) # energy probe seems to be not available for now
    # profiler.activity_probe()
    # profiler.memory_probe()

    # Start runtime, kinda jank
    sg.run(condition=RunSteps(num_steps=1), run_cfg=run_config)
    # TODO: Reset all voltage / current / spike readout
    output_sink.data.set(np.zeros_like(output_sink.data.get()))
    
    times = []
    
    correct = 0
    total = 0
    
    counter = NUM_SAMPLES
    for data in tqdm(test_set_loader):
        if counter == 0:
            break
        counter -= 1
    
        if type(data) is not tuple:
            data = tuple(data)
    
        data, label = S2S(data) # comes out with shape (batch, timestep, feature)
        data = data.squeeze()
        data = data.transpose(1, 0) # convert to shape (feature, timestep)
        data = data.detach().numpy()
    
        assert(data.shape == (NUM_FEATURES, NUM_TIMESTEPS))
    
        sg.data.set(data)
    
        sg.run(condition=RunSteps(num_steps=NUM_TIMESTEPS), run_cfg=run_config)
                
        # End of batch: Add to results, reset network back to zeros
        spikes = output_sink.data.get()
        output_sink.data.set(np.zeros_like(spikes))
        # TODO: reset the rest of the network
        
        # spikes are in shape (NUM_CLASSES, NUM_TIMESTEPS)
        pred = choose_max_count(torch.tensor(spikes).transpose(0,1).unsqueeze(dim=0))
        if pred == label.item():
            correct += 1
        total += 1
        
    sg.stop()
    
    # # Performance profiling
    # total_time = profiler.execution_time.sum()
    # times.append(total_time)
    
    # TODO: calculate performance metrics

    # Accuracy
    accuracy = correct / total
    print("Accuracy:", accuracy)