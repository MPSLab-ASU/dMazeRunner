dMazeRunner can be used to perform design space explroation for finding efficient designs.
This experiment aims at tuning the memory sizes for a 256-PE accelerator design that is evaluated on various convolution layers of ResNet model.

The script tune_memory.py can be executed as follows:

  * Step 1: Evaluation Mode: 

    `python tune_memory.py run <layer_index>`

  * Step 2: Parsing Mode (for producing output files):

    `python tune_memory.py parse <layer_index>`

    Note that a number `<layer_index>` indicates the various ResNet layers to be evaluated. For example,
    
    0 : ResNet `Conv1`  
    1 : ResNet `Conv2_2`  
    2 : ResNet `Conv3_2`   
    3 : ResNet `Conv4_2`   
    4 : ResNet `Conv5_1`    
    5 : ResNet `Conv5_2`
    

Output excel files provide stats such as - design points `(SPM_Size, RF_Size, Total_PEs)`, EDP, Total mappings evaluated for a design point, and time taken to evaluate a design point. 

Please note that EDP is obtained through rapid search exploration heuristics of dMazeRunner (i.e., with search optimization techniques 1--3). For more details about these search optimization heuristics, please see `<dMazeRunner>/scripts/run_optimizer.py` or `<dMazeRunner>/experiments/rapid_search_space_exploration`. 
