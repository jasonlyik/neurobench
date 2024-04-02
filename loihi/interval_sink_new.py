import numpy as np
import typing as ty
import random
import time

from lava.magma.core.process.variable import Var
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, RefPort, OutPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyRefPort

def choose_greatest(spike_count):
    # max_count, except that when multiple are max, it will randomly choose the winner
    greatest_indexes = np.argwhere(spike_count == np.amax(spike_count)).flatten().tolist()
    return random.choice(greatest_indexes)

class IntervalReadAccuracy(AbstractProcess):
    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 interval: int = 1,
                 offset: int = 0) -> None:
        super().__init__(shape=shape,
                         interval=interval, offset=offset)
        self.interval = Var((1,), interval)
        self.offset = Var((1,), offset % interval)
        
        self.inp_pred = InPort(shape=shape)
        self.inp_label = InPort(shape=(1,))
        
        self.label = Var((1,), 0)

        self.spike_count_old = Var(shape=shape, init=np.zeros(shape))

        self.correct = Var((1,), 0)
        self.total = Var((1,), 0)

@implements(proc=IntervalReadAccuracy, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class IntervalReadAccuracyModel(PyLoihiProcessModel):
    """Abstract ring buffer receive process model."""
    interval: np.ndarray = LavaPyType(np.ndarray, int)
    offset: np.ndarray = LavaPyType(np.ndarray, int)

    inp_pred: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    inp_label: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    label: np.ndarray = LavaPyType(np.ndarray, np.int32)

    spike_count_old: np.ndarray = LavaPyType(np.ndarray, np.int32)

    correct: np.ndarray = LavaPyType(np.ndarray, np.int32)
    total: np.ndarray = LavaPyType(np.ndarray, np.int32)
    
    def run_spk(self) -> None:
        # The label is sent every timestep by the SpikeDataloader so it must be received or else the buffer fills up.
        self.label = self.inp_label.recv()

    def post_guard(self) -> None:
        condition = (self.time_step - 1) % self.interval == self.offset
        return condition

    def run_post_mgmt(self) -> None:
        spike_count = self.inp_pred.recv()
        sample_spikes = spike_count.copy()
        sample_spikes -= self.spike_count_old
        self.spike_count_old = spike_count.copy()

        pred = choose_greatest(sample_spikes)

        self.correct += (pred == self.label.item())
        self.total += 1

        if (self.total % 100) == 0:
            print("Life check, sample", self.total)
        