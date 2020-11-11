# dMazeRunner
dMazeRunner is a framework for automated and efficient search-space and design-space exploration for executing loop kernels on coarse-grained programmable dataflow accelerators. dMazeRunner employs analytical model of loop executions on dataflow accelerator architecture, helping guide search-space optimizations for mapping the loops. Through dMazeRunner, users can optimize dataflow acceleration of various ML models, analyze optimized mappings, and explore the mapping search-space and the design-space of accelerator architectures. 



## Installation
1. clone this repository with `--recursive` option. 

    ```bash
    git clone --recursive <repository-url>
    ```

2. Build tvm: Follow instructions about [building tvm from source](https://docs.tvm.ai/install/from_source.html).

3. Include `path/to/dMazeRunner/python` in the environment variable `PYTHONPATH`.

    ```bash
    export PYTHONPATH=/path/to/dMazeRunner/python:$PYTHONPATH
    ```
4. Ensure that you have installed necessary dependencies specified in the file `pip-dependencies.txt`. You may install them in your python virtual environment.

    ```bash
    pip3 install -r pip-dependencies.txt
    ```
    
For additional information, please refer to file `setup.txt`.



## High-level Overview

<p align="center">
  <img src="https://labs.engineering.asu.edu/mps-lab/wp-content/uploads/sites/8/2019/09/dMazeRunner.png"/ height="300">
</p>
    
1. **Front-end**: Front-end of the framework deals with parsing the specified deep learning model and extracts the necessary information about target loop-nests. The front-end leverages TVM environment to support multiple ML libraries such as MXNet and keras. 

2. **Generating effective mapping space**: After analyzing a loop-nest, dMazeRunner formulates a holistic representation which features explicit tiling of the loops for their spatial execution on PEs as well as for their temporal execution (i.e., for accessing data from Register Files of PEs, shared ScratchPad-Memory, and DRAM). Formulated mapping space exhibit all solutions that are valid (subjected to architectural/loop-functionality/tiling constraints) and feature unique data reuse cost, capturing a vast space of execution methods.

3. **Search-space optimizations**: To facilitate analysis and optimizations of dataflow acceleration for domain non-experts, dMazeRunner employs a few search-reduction strategies. Through auto-optimizer, users can achieve optimized dataflow execution methods for various layers or even entire model in a few seconds. The framework outputs the execution methods which are optimized for performance/energy/energy-delay-product, along with the corresponding estimates of the execution metrics. Implementation with multi-threading and caching of the commonly invoked routines enable faster explorations of the mappings at both layer-level and model-level (in order of a few seconds).  

4. **Analytical modeling of dataflow execution**: To determine the goodness of an execution method statically, dMazeRunner explicitly models computation and communication costs for various architectural features and estimates the execution time and energy consumption. From an input loop-nest, the model analyzes indexing expressions and data dependencies of operands. Then, the model determines various data reuse factors, DMA invocations and burst sizes for managing non-contiguous data in SPM, miss penalty, data communication through NoC, and any stall cycles inter/intra-PE-group communication.
    
If you are interested in learning about — dataflow accelerator architectures, mapping the loops on dataflow accelerators, analyzing dataflow execution, drastic pruning of the mapping search-space, etc., more details are in dMazeRunner paper.



## Exploring Dataflow Accelerations of Models Through dMazeRunner

Script `scripts/run_optimizer.py` shows how to download and optimize the models from popular frontends. The script outputs the execution methods which are optimized for performance/energy/energy-delay-product, along with the corresponding estimates of the execution metrics.

For example, users can optimize dataflow acceleration of ResNet model with the following command:

```bash
python run_optimizer.py --frontend mxnet --model resnet18_v1 --auto-optimize
```

For addiotional details, please refer to `scripts/README`. More information about the execution methods is available in `scripts/execution_methods.txt`. Please note that currently dMazeRunner supports analysis and optimizations for a limited MXNet and keras models. In near future, we plan to enable front-end support for more ML application libraries, as well as improvisations in analytical model and more sophisticated mapping-search-space and design-space explorations.



## Publication

If you find dMazeRunner useful or if you use it in your work, please cite the following paper:

Shail Dave, Youngbin Kim, Sasikanth Avancha, Kyoungwoo Lee, Aviral Shrivastava, dMazeRunner: Executing Perfectly Nested Loops on Dataflow Accelerators, in ACM Transactions on Embedded Computing Systems (TECS) \[Special Issue on ESWEEK 2019 - ACM/IEEE International Conference on Hardware/Software Codesign and System Synthesis (CODES+ISSS)\].
\[[PDF](https://dl.acm.org/doi/pdf/10.1145/3358198)\] \[[Slides](https://mpslab-asu.github.io/publications/slides/Dave2019TECS.pptx)\]

(Key topics: Understanding spatiotemporal execution and dataflow mechanisms; Abstraction for comprehensive search-space, Analytical cost model for hardware accelerators; Determining unique and maximum data reuse of tensors; How to optimize mappings for quick search-space exploration.)

```
@article{dave2019dmazerunner,
  title={Dmazerunner: Executing perfectly nested loops on dataflow accelerators},
  author={Dave, Shail and Kim, Youngbin and Avancha, Sasikanth and Lee, Kyoungwoo and Shrivastava, Aviral},
  journal={ACM Transactions on Embedded Computing Systems (TECS)},
  volume={18},
  number={5s},
  pages={1--27},
  year={2019},
  publisher={ACM New York, NY, USA}
}   
```

Shail Dave, Aviral Shrivastava, Youngbin Kim, Sasikanth Avancha, Kyoungwoo Lee, dMazeRunner: Optimizing Convolutions on Dataflow Accelerators, in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020. \[[PDF](https://mpslab-asu.github.io/publications/papers/Dave2020ICASSP.pdf)\] \[[Slides](https://mpslab-asu.github.io/publications/slides/Dave2020ICASSP.pptx)\] \[[Talk](https://www.youtube.com/watch?v=21F79Taelts)\]

(Key topics: Tool capabilities including allowing users to determine efficiency of execution methods, automated and quick optimizations for mapping DNN operators onto accelerators, design space exploration.)

```
@inproceedings{dave2020dmazerunner,
  title={dMazeRunner: Optimizing Convolutions on Dataflow Accelerators},
  author={Dave, Shail and Shrivastava, Aviral and Kim, Youngbin and Avancha, Sasikanth and Lee, Kyoungwoo},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1544--1548},
  year={2020},
  organization={IEEE}
}
```

### Contact Information

For any questions or comments on dMazeRunner, please email us at cmlasu@gmail.com
