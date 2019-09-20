This experiment validates energy consumption of dataflow execution for ResNet layers. It covers various dataflow mechanisms which represent how different loops are executed spatially. For example, `Fx|Fy` represents a weight stationary mechanism where PEs are grouped based on unrolling `Fy` and `Fx` loops for spatial execution. Similarly, `Ox|Oy` represents output stationary mechanism. Note that execution methods for various dataflow mechanisms also exhibit the variations in temporal execution (different data reuse patterns) and the spatial execution of more than two loops. Execution methods for evaluations on dMazeRunner were obtained from the energy optimizer [1]; target accelerator featured 16x16 PE-array with 512B register file (RF) per processing element (PE) and a shared 128 kB scratch-pad memory (SPM). Experiment can be executed with following command:

`python run_expr.py`

Generated outputs are available in out.xlsx, which includes energy estimation for the accelerator, including energy break-down for accessing system resources like register files, scratch-pad, interconnect, PE computation, and DRAM.

Reference:

[1] Xuan Yang et al. Dnn energy model and optimizer. https://github.com/xuanyoya/CNN-blocking/tree/dev.
