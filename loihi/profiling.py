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
    
    ## From pilotnet benchmark tutorial: https://lava-nc.org/lava-lib-dl/netx/notebooks/pilotnet_snn/benchmark.html
    # power_logger = loihi2_profiler.Loihi2Power(num_steps=10)
    # runtime_logger = loihi2_profiler.Loihi2ExecutionTime()
    # memory_logger = loihi2_profiler.Loihi2Memory()
    # activity_logger = loihi2_profiler.Loihi2Activity()
    
    # run_config = Loihi2HwCfg(callback_fxs=[power_logger, runtime_logger, memory_logger, activity_logger])
    
    run_config = Loihi2HwCfg()
    
    from lava.utils.loihi2_profiler_api import Loihi2HWProfiler
    profiler = Loihi2HWProfiler(run_config)
    profiler.execution_time_probe(num_steps=num_steps, t_start=1, t_end=num_steps, dt=1, buffer_size=1024)
    # profiler.energy_probe(num_steps=num_steps) # energy probe seems to be not available for now
    profiler.activity_probe()
    profiler.memory_probe()
    
    lif1.run(condition=RunSteps(num_steps=10), run_cfg=run_config)
    
    lif1.stop()
    
    # Mapping information
    # print(profiler.num_alloc_chips)
    # print(profiler.num_available_chips)
    # print(profiler.statement)
    
    # Execution Time
    execution_time = profiler.execution_time
    host_time = profiler.host_time
    learning_time = profiler.learning_time
    management_time = profiler.management_time
    pre_lrn_time = profiler.pre_lrn_mgtm_time
    spiking_time = profiler.spiking_time
    
    # print(execution_time)
    # print(host_time)
    # print(learning_time)
    # print(management_time)
    # print(pre_lrn_time)
    # print(spiking_time)
    
    # Energy
    # avg_power = profiler.power
    # static_power = profiler.static_power
    # dynamic_power = profiler.dynamic_power
    # vdd_power = profiler.vdd_power # computing logic (neurocore, mesh, routers, ...)
    # vddm_power = profiler.vddm_power # SRAM power
    # vddio_power = profiler.vddio_power # FPIO/PIO interface power
    # total_energy = profiler.energy
    # static_energy = profiler.static_energy
    # dynamic_energy = profiler.dynamic_energy
    
    
    # Activity
    # API does not get any activity metrics, use lava.utils.loihi2_profiler.Loihi2Activity if needed
    
    # Memory
    # API does not get any memory metrics, use lava.utils.loihi2_profiler.Loihi2Memory if needed
    
    # plotting time
    profiler.plot_execution_time("./execution_plot.png")
    
    # plotting activity
    profiler.plot_activity("./activity_plot.png")
    
    # plotting memory
    profiler.plot_memory_util("./memory_plot.png")