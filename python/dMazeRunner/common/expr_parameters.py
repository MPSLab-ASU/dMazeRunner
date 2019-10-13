# Parameters used by analytical model
# Requires details about accelerator architecture execution
class Environment:
    def __init__(self,
                 # Pipelining of the processing elements
                 # Each PE features a pipelined multiply-and-accumulate unit
                 # and a register file.
                 pe_pipelined=True,
                 pe_pipeline_stages=1,
                 data_type_size_in_bytes=2,
                 # Direct Memory Access Controller for DRAM accesses
                 dma_init_cycles=291,
                 dma_cycles_per_byte=0.24,
                 dma_frequency=3200,     # (MHz)
                 # Accelerator frequency in MHz
                 dataflow_accel_frequency=500,
                 # Energy consumption of architectural resources (in pJ)
                 # Following corresponds to 28 nm technology node.
                 # Possible to provide normalized energy values
                 # (e.g., values normalized to MAC operation)
                 rf_energy=0.96,
                 communication_energy=2,
                 spm_energy=13.5,
                 dram_energy=200,
                 mac_energy=0.075
                 ):
        self.pe_pipelined = pe_pipelined
        self.pe_pipeline_stages = pe_pipeline_stages
        self.data_size = data_type_size_in_bytes
        self.dma_init_cycles = dma_init_cycles
        self.dma_cycles_per_byte = dma_cycles_per_byte
        self.dma_frequency = dma_frequency
        self.dataflow_accel_frequency = dataflow_accel_frequency
        self.energy_cost = (rf_energy, communication_energy, spm_energy, dram_energy, mac_energy)


# Parameters used by the map-space generator and optimizer
# Requires basic info about architecture spec
class ExprParameters:
    def __init__(self, _env=None,
                    # Total number of Processing Elements
                    CGRA_SIZE = 256,
                    # size of Register File per PE (in bytes)
                    RF_SIZE = 512,
                    # All operands are treated of same size
                    BYTES_PER_DATA = 2,
                    # Multi-bank on-chip scratchpad is shared among PEs (bytes).
                    # Scratchpad memory is double-bufferred for hiding
                    # memory access latency.
                    TOTAL_BANKS_IN_SPM = 32,
                    TOTAL_BYTES_IN_BANK = 2048,
                    # Parameters used by mapping-space generator and optimizer
                    # Utilization factors for architectural resources
                    PE_UTILIZATION = 0,
                    RF_UTILIZATION = 0,
                    SPM_UTILIZATION = 0,
                    # Parameters for optimization strategies of search-reduction
                    PRUNE_NO_FEATURE_DRAM = False,
                    PRUNE_NO_REDUCTION = False,
                    MIN_EXEC_METHODS = 1,
                    THRESHOLD_BEGIN_REDUCE_SPM_UTIL = 0.6,
                    THRESHOLD_DISABLE_OPTS = 0.3
                    ):
        self.CGRA_SIZE = CGRA_SIZE
        self.RF_SIZE = RF_SIZE / BYTES_PER_DATA
        self.BYTES_PER_DATA = BYTES_PER_DATA
        self.TOTAL_BANKS_IN_SPM = TOTAL_BANKS_IN_SPM
        self.TOTAL_BYTES_IN_BANK = TOTAL_BYTES_IN_BANK
        self.PE_UTILIZATION = PE_UTILIZATION
        self.RF_UTILIZATION = RF_UTILIZATION
        self.SPM_UTILIZATION = SPM_UTILIZATION
        self.PRUNE_NO_FEATURE_DRAM = PRUNE_NO_FEATURE_DRAM
        self.PRUNE_NO_REDUCTION = PRUNE_NO_REDUCTION
        self.MIN_EXEC_METHODS = MIN_EXEC_METHODS
        self.THRESHOLD_BEGIN_REDUCE_SPM_UTIL = THRESHOLD_BEGIN_REDUCE_SPM_UTIL
        self.THRESHOLD_DISABLE_OPTS = THRESHOLD_DISABLE_OPTS

        if _env != None:
            self.env = _env
        else:
            self.env = Environment()
