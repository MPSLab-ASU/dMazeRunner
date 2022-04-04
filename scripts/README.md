## Optimizing Model Executions for Dataflow Acceleration

dMazeRunner can explore optimized mappings for the execution of individual layers of a specified model, as well as for the entire model. For a given dataflow accelerator architecture (described in `arch_spec.json`), dMazeRunner can analyze the mapping search-space and estimates execution metrics for various *execution methods*. Each execution method refers to tiling of the loops for processing different data onto PEs and accessing the data from memories (Register Files of Processing Elements, shared Scratch-Pad Memory, and DRAM), as well as ordering of the loops. Execution methods yield different dataflows in spatiotemporal execution.  

Script `run_optimizer.py` shows how to download and optimize the models from popular frontends. The script outputs the execution methods which are optimized for performance/energy/energy-delay-product, along with the corresponding estimates of the execution metrics. Implementation with multi-threading and caching of the commonly invoked routines enable faster explorations of the mappings at both layer-level and model-level (in order of a few seconds).

* For example, users can optimize dataflow acceleration of AlexNet model with the following command:

```bash
python run_optimizer.py --frontend mxnet --model alexnet --auto-optimize
```

* Similarly, Conv5_2 layer of ResNet model can be auto-optimized as:

```bash
python run_optimizer.py --frontend mxnet --model resnet18_v1 --layer-index 62 --arch-spec arch_spec.json --auto-optimize
```

* Users can also specify custom models through .json files. Directory "models" provides examples for some models for computer vision and language processing tasks.

```bash
python run_optimizer.py --files-model-spec ../models/bert-base-uncased.json --arch-spec arch_spec.json --auto-optimize
```

* Instead of auto optimization, it is also possible to explore the impact of individual optimization strategy, or enabling additional strategies for a directed search. For example,

```bash
python run_optimizer.py --frontend mxnet --model resnet18_v1 --layer-index 62 --opt-min-utilization 0.8 0.8 0.5 --opt-no-feature-dram --opt-no-spatial-reduction 
```

* For more details about optimization options and supported models, please run `python run_optimizer.py --help`.
  
Interpretation of execution methods is provided in the file `execution_methods.txt`. Moreover, information about architecture specifications is available in the file `</path/to/dMazeRunner/python/dMazeRunner/common/expr_parameters.py>`.
  

Please note that dMazeRunner targets only conv and gemm layers, since they are often performance-critical and typically accelerated onto PEs of the accelerator, while functions like non-linear activation or batch-normalization etc. are either performed through special function units of the accelerator, or immediately onto PEs durnig writing-back the output of the convolution. Thus, optimized execution methods for the specified model corresponds to conv and gemm layers.


## Design Space Exploration

Since dMazeRunner employs analytical model for accelerating loop-nests on dataflow accelerator architecture and allows quick search-explorations, it can be used for exploring efficient architecture designs. For example, script `dse.py` shows an example of how to explore the implications of different designs (with varying sizes of Register Files, ScratchPad Memory, and total PEs) for ResNet layers. It considers design variations specified in `dse_parameters.py` and evaluates several designs by finding optimized execution methods and provides estimates of the execution metrics for the designs. 

Example script for DSE can be invoked as:

* Step 1: Evaluation Mode: Find different designs and optimized mappings
  ```bash
  python dse.py find_config --layer-index 5 --config-file test
  ```
* Step 2: Parsing Mode (produce output files)
  ```bash
  python dse.py run_dse     --layer-index 5 --config-file test
  ```

* number `<layer_index>` indicates the various ResNet layers to be evaluated. For example,
    
  0 : ResNet `Conv1`  
  1 : ResNet `Conv2_2`  
  2 : ResNet `Conv3_2`   
  3 : ResNet `Conv4_2`   
  4 : ResNet `Conv5_1`    
  5 : ResNet `Conv5_2`
    

Output excel files provide stats such as - design points `(SPM_Size, RF_Size, Total_PEs)`, EDP, total mappings evaluated for a design point, and time taken to evaluate a design point. For more details about the script, please run `python dse.py --help`. Please note that this script shows an example for performing DSE on ResNet layers. Simillarly, `experiments/dse_tune_memory_256_PE` shows tuning of the memory sizes for a 256-PE accelerator design. We plan to integrate the DSE support soon with our front-end so that design optimizations for different models can be automatically analyzed. 
