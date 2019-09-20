### Examples

dMazeRunner can explore optimized mappings for the execution of individual layers of a specified model, as well as for the entire model. For a given dataflow accelerator architecture (described in `arch_spec.json`), dMazeRunner can analyze the mapping search-space and estimates execution metrics for various *execution methods*. Each execution method refers to tiling of the loops for processing different data onto PEs and accessing the data from memories (Register Files of Processing Elements, shared Scratch-Pad Memory, and DRAM), as well as ordering of the loops. Execution methods yield different dataflows in spatiotemporal execution.  

These examples show how to set up specific configurations for conv/gemm layers, architecture specifcations, and execution methods.  

* An example of execution method for ResNet_Conv_5_2 layer

```bash
python conv.py [--arch-spec arch_spec.json]
```

* An example of execution method for dense matrix multiplication (gemm) for matrices of size 1000x1000.

```bash
python gemm.py [--arch-spec arch_spec.json]
```

The output provides estimation of the execution metrics - energy, execution cycles, and energy-delay-product.Using these examples, one can set-up different configurations for layer, architecture, execution methods, etc., and analyze their impact. More information about interpretation of execution methods is provided in the file `scripts/execution_methods.txt`. Moreover, information about architecture specifications is available in the file `</path/to/dMazeRunner/python/dMazeRunner/common/expr_parameters.py>`. Dataflow accelerator architecture can be specified using `arch_spec.json`, and when unspecified, the analytical model is provided a default architecture setup. 

Please note that dMazeRunner targets only conv and gemm layers, since they are often performance-critical and typically accelerated onto PEs of the accelerator, while functions like non-linear activation or batch-normalization etc. are either performed through special function units of the accelerator, or immediately onto PEs durnig writing-back the output of the convolution. Thus, optimized execution methods for the specified model corresponds to conv and gemm layers.

For additional information about how to download and optimize the models/layers from popular frontends for dataflow acceleration, please refer to the script scripts/`run_optimizer.py`. 
