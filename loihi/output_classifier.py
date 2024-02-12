import os
import numpy as np
import typing as ty
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

from numpy import genfromtxt

# Import parent classes for ProcessModels
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.model import PyLoihiProcessModel

# Import ProcessModel ports, data-types
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

# Import execution protocol and hardware resources
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU

# Import decorators
from lava.magma.core.decorator import implements, requires

class OutputClassifier(AbstractProcess):
    """Process to gather spikes from 10 output LIF neurons and interpret the
    highest spiking rate as the classifier output"""

    def __init__(self, num_classes: ty.Optional[int] = 35, num_steps_per_sample: ty.Optional[int] = 201):
        super().__init__()
        shape = (num_classes,)
        self.spikes_in = InPort(shape=shape)
        self.spikes_accum = Var(shape=shape)  # Accumulated spikes for classification
        self.num_steps_per_sample = Var(shape=(1,), init=num_steps_per_sample)
        self.pred_label = Var(shape=(1,), init=0)

@implements(proc=OutputClassifier, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputClassifierModel(PyLoihiProcessModel):
    spikes_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spikes_accum: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
    num_steps_per_sample: int = LavaPyType(int, int, precision=32)
    pred_label: int = LavaPyType(int, int, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.current_img_id = 0

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        spk_in = self.spikes_in.recv()
        self.spikes_accum = self.spikes_accum + spk_in
