from tqdm import tqdm
import numpy as np
import torch

from lava.lib.dl import netx
from lava.lib.dl import slayer

from neurobench.datasets import SpeechCommands
from neurobench.preprocessing import S2SPreProcessor
from neurobench.postprocessing import choose_max_count

from torch.utils.data import DataLoader

from lava.proc.io.source import RingBuffer as SpikeGenerator
from lava.proc.io.dataloader import SpikeDataloader
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.io.sink import RingBuffer as Sink

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg

# This works for the NeuroBench S2S, which has 201/20 timesteps/features.
NUM_TIMESTEPS = 201 # timesteps for S2S GSC sample
NUM_FEATURES = 20 # features of S2S GSC sample
NUM_CLASSES = 35

class S2S_GSC_DataLoader():
    # DataLoader to feed SpikeDataloader with pre-processed GSC samples
    def __init__(self):
        self.dataset = SpeechCommands(path="./data/speech_commands/", subset="testing")
        self.s2s = S2SPreProcessor()

    def __getitem__(self, idx):
        data, label = self.dataset[idx] # data = (timestep, feature)
        data = data.unsqueeze(0) # (1, timestep, feature)
        data, label = self.s2s((data, label)) # (1, timestep, feature)
        data = data.squeeze() # (timestep, feature)
        data = data.transpose(1, 0) # (feature, timestep)
        data = data.detach().numpy()
        return data, label.item()

    def __len__(self):
        return len(self.dataset)

# argparse for trained network path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--net", type=str, required=True)
args = parser.parse_args()

# load trained network
net = netx.hdf5.Network(net_config=args.net, input_message_bits=2)

# preprocessor
s2s_gsc = S2S_GSC_DataLoader()
s2s_gsc_loader = DataLoader(s2s_gsc, batch_size=1, shuffle=False)

# Lava-Loihi units
run_config = Loihi2SimCfg(select_tag="fixed_pt")

sg = SpikeGenerator(data=s2s_gsc[0][0])

output_sink = Sink(shape=(NUM_CLASSES, ), buffer=NUM_TIMESTEPS)

sg.s_out.connect(net.inp)
net.out.connect(output_sink.a_in)

# start runtime so can set data
net.run(condition=RunSteps(num_steps=NUM_TIMESTEPS), run_cfg=run_config)

correct = 0
total = 0

print("Starting run.")
for data, label in tqdm(s2s_gsc_loader):
    data = data.squeeze()
    data = data.detach().numpy()
    assert(data.shape == (NUM_FEATURES, NUM_TIMESTEPS))
    sg.data.set(data)

    net.run(condition=RunSteps(num_steps=NUM_TIMESTEPS), run_cfg=run_config)            
    
    
    spikes = output_sink.data.get()
    output_sink.data.set(np.zeros_like(spikes))

    # Reset all network state back to zero
    # each block is a Dense connected to LIF
    for block in net.layers:
        block.neuron.u.set(np.zeros_like(block.neuron.u.get()))
        block.neuron.v.set(np.zeros_like(block.neuron.v.get()))
    
    # spikes are in shape (NUM_CLASSES, NUM_TIMESTEPS)
    pred = choose_max_count(torch.tensor(spikes).transpose(0,1).unsqueeze(dim=0))

    # print("Predicted:", pred.item(), "Label:", label.item())
    if pred == label.item():
        correct += 1
    total += 1


net.stop()

print("Accuracy:", correct / total)