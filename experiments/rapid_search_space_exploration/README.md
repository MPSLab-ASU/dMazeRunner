dMazeRunner can perform rapid design space explroation for finding efficient mappings for the specified description of a dataflow accelerator architecture. This experiment targets a 256-PE accelerator design with 512B RF per PE and 128 kB shared scratchpad (double-buffered). It shows that user can specify optimizations for a directed exploration. In this experiment, mappings are explored for various convolution layers of ResNet-18 model.

The script run_optimizer.py can be executed as follows:

`python run_optimizer.py --layer-index=<layer_index> [optimization options]`


Specifiying optimization options for a directed search are optional. For this experiment, supported optimizations are:

  `--opt-utilization`          : Prune execution methods featuring low utilizations of RF, SPM, and PEs. (OPT1)

  `--opt-no-feature-dram`      : Prune execution methods that access non-contiguous 2D features from DRAM. (OPT2)

  `--opt-no-spatial-reduction` : Prune execution methods that require inter-PE communication for reduction. (OPT3)

Each of these optimization strategies can be specified stand-alone or can be combined together. For example, the command specified below will explore mappings for ResNet layer conv5_2 with all the three search reduction strategies.

`python run_optimizer.py --layer-index=5 --opt-utilization --opt-no-feature-dram --opt-no-spatial-reduction`

Note that a number `<layer_index>` indicates the various ResNet layers to be evaluated. For example,
    
  0 : ResNet `Conv1`  
  1 : ResNet `Conv2_2`  
  2 : ResNet `Conv3_2`   
  3 : ResNet `Conv4_2`   
  4 : ResNet `Conv5_1`    
  5 : ResNet `Conv5_2`

Output provide stats such as - optimized Energy-Delay Product (EDP), execution method that yielded minimum EDP, optimized energy, execution method that yielded optimized energy, total execution methods evaluated, and time taken to evaluate a convolution layer.
