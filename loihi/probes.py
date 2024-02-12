from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg

from lava.utils import loihi2_profiler

# Instantiate Lava processes to build network
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF
import numpy as np

if __name__ == "__main__":
    num_steps = 10
    
    lif1 = LIF(shape=(1, ))
    dense = Dense(weights=np.random.rand(1, 1))
    lif2 = LIF(shape=(1, ))
    
    # Connect processes via their directional input and output ports
    lif1.out_ports.s_out.connect(dense.in_ports.s_in)
    dense.out_ports.a_out.connect(lif2.in_ports.a_in)
    
    from lava.utils.loihi2_profiler import Loihi2Activity
    from lava.utils.loihi2_profiler import Loihi2ExecutionTime
    from lava.utils.loihi2_profiler import Loihi2Memory
    from lava.utils.loihi2_profiler import Loihi2Power
    
    # From pilotnet benchmark tutorial: https://lava-nc.org/lava-lib-dl/netx/notebooks/pilotnet_snn/benchmark.html
    # power_logger = loihi2_profiler.Loihi2Power(num_steps=10)
    runtime_logger = loihi2_profiler.Loihi2ExecutionTime()
    memory_logger = loihi2_profiler.Loihi2Memory()
    activity_logger = loihi2_profiler.Loihi2Activity()
    
    # run_config = Loihi2HwCfg(callback_fxs=[power_logger, runtime_logger, memory_logger, activity_logger])
    run_config = Loihi2HwCfg(callback_fxs=[runtime_logger, memory_logger, activity_logger])
    
    lif1.run(condition=RunSteps(num_steps=10), run_cfg=run_config)
    lif1.stop()
    runtime_logger = loihi2_profiler.Loihi2ExecutionTime()
    memory_logger = loihi2_profiler.Loihi2Memory()
    activity_logger = loihi2_profiler.Loihi2Activity()
    run_config = Loihi2HwCfg(callback_fxs=[runtime_logger, memory_logger, activity_logger])
    lif1.run(condition=RunSteps(num_steps=10), run_cfg=run_config)
    lif1.stop()
    
    
    # # Execution Time
    # avg_time = runtime_logger.avg_time_per_step
    # execution_time = runtime_logger.execution_time_per_step
    # spiking_time = runtime_logger.spiking_time_per_step
    # management_time = runtime_logger.management_time_per_step
    # host_time = runtime_logger.host_time_per_step
    # learning_time = runtime_logger.learning_time_per_step
    # pre_lrn_time = runtime_logger.pre_lrn_time_per_step
    
    
    # # Activity
    # activity_core_idx = activity_logger.core_idx
    # spikes_in = activity_logger.spikes_in
    # syn_ops = activity_logger.syn_ops
    # updates = activity_logger.dendrite_updates
    # spikes_out = activity_logger.axon_out
    
    # # Memory
    # memory_core_idx = memory_logger.core_idx
    # small_mpds = memory_logger.small_mpds
    # large_mpds = memory_logger.large_mpds
    # total_mpds = memory_logger.total_mpds
    
    # breakpoint()