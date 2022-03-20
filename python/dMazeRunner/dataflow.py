
from __future__ import absolute_import

import tvm

import pprint
import json
import logging
from functools import reduce
from itertools import combinations
from itertools import chain
from collections import OrderedDict
from functools import lru_cache
import math
import operator

from nnvm.compiler import build_module

from dMazeRunner.common import expr_parameters

log = logging.getLogger("dataflow")

class Layer:
    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.loops = []

    def __str__(self):
        return "index: {}, name: {}".format(self.index, self.name)


class LayerBase:
    def __init__(self, env, **kargs):
        self.env = env
        self.name = kargs["name"]
        self.index = kargs["index"] if "index" in kargs else 1
        self.instances = kargs["instances"] if "instances" in kargs else 1
        self.tiling_levels = ["DRAM", "SPM", "RF", "Spatial"]

        # Child class should initialize these properties
        self.base_TCs = None
        self._loop_TCs = None
        self._loop_IVs = None
        self._default_loop_order = None
        self.loop = None
        self._loop_orderings = None
        self._tensors = None #dict mapping tensor name and its instance


    def _get_num_different_pixels(self, op, idx_list):
        index_expr_string = op.index.__repr__()
        local_vars = {}
        for idx_name, idx_value in zip(self._default_loop_order, idx_list):
            local_vars[idx_name.lower()] = idx_value
        evaluated = eval(index_expr_string, {}, local_vars)
        return evaluated


    def _get_tensor_from_name(self, name):
        return self._tensors[name]


    def __repr__(self):
        raise NotImplementedError()


    # Obtain Trip-Counts (TCs) of a loop index variable (IV) for a tiling level
    # Valid options are: spatial, and memory levels (RF, SPM, DRAM) and
    # loop (i.e., baseline trip-counts provided by bounds of loops in a kernel)
    # For example, get_TripCounts(‘ox’, ‘Spatial’) will search for index ‘ox’
    # in the list of all index variables of a kernel and, the function will
    # return the value of iteration counts Ox_SPATIAL.
    def get_TripCounts(self, iv, option):
        if option == "loop" and iv in self.base_TCs:
            return self.base_TCs[iv]
        assert(option in self.tiling_levels)
        assert(iv in self.base_TCs)
        iv_name = iv + "_" + option
        return self._loop_TCs[iv_name]


    # Generate collective trip-counts of all loops at a tiling level.
    # For example, if each of the two loops with index variables x and y iterate
    # for 10 iterations, then for option=loop, the function will return value
    # X*Y = 10*10 = 100. Similarly, for X_RF = Y_RF = 2, the function will
    # return X_RF * Y_RF = 4.
    @lru_cache(maxsize=None)
    def get_TripCounts_all_IVs(self, option):
        mult_func = lambda x, y: x*y
        if option == "loop":
            return reduce(mult_func, self.base_TCs.values())
        else:
            assert(option in self.tiling_levels)
            IVs = self._loop_IVs[option]
            TCs = [ self._loop_TCs[iv] for iv in IVs ]
            return reduce(mult_func, TCs)


    # Get energy consumption of system components (used for analytical model).
    # For a given technology node (e.g. 14 nm), these energy values can be
    # either normalized (to MAC operation) or can be absolute values (in pJ).
    # For example, get_Energy(0) will return energy[RF] = 0.96 pJ.
    # TODO: Obtain energy values from parsing the architecture description.
    @lru_cache(maxsize=None)
    def get_energy(self, idx):
        if type(idx) == int:
            return Dataflow.energy_cost[idx]
        if type(idx) == str:
            index = {
                "RF": 0,        # RF access (local memory to PEs)
                "network": 1,   # Inter-PE/Multicast Communication to PE
                "SPM": 2,       # Accessing on-chip scratch-pad memory (shared)
                "DRAM": 3,      # Cost for accessing data from DRAM
                "MAC": 4        # Computing Multiply-and-ACcumualate operations
            }
            return self.env.energy_cost[index[idx]]
            #return Dataflow.energy_cost[index[idx]]
        assert(False)


    # In current dataflow model, all the data elements get eventually accessed
    # from RFs of PEs and therefore, regardless of execution of loops spatially
    # and temporally, each operation is always accessed from RF.
    @lru_cache(maxsize=None)
    def computeEnergyCost_RF(self):
        total_loop_operations = self.get_TripCounts_all_IVs("loop")
        reads, writes = self._get_reads_writes()
        RF_read_cost = len(reads) * total_loop_operations * self.get_energy("RF")
        RF_write_cost = len(writes) * total_loop_operations * self.get_energy("RF")
        return RF_read_cost + RF_write_cost

        write_op = writes[0]
        index_vars = self._get_index_vars(write_op)
        other_vars = list(set(self._default_loop_order) - set(index_vars))
        index_tcs = [ self.get_TripCounts(iv, "RF") for iv in index_vars ]
        other_tcs = [ self.get_TripCounts(iv, "RF") for iv in other_vars ]
        a = reduce(operator.mul, index_tcs)
        b = reduce(operator.mul, other_tcs)
        loop_operations = a * (b-1)
        RF_write_cost = len(writes) * loop_operations * self.get_energy("RF")
        return RF_read_cost + RF_write_cost


    @lru_cache(maxsize=None)
    def _get_reads_writes(self):
        stores = self._get_stores()

        def make_find_read_write(read_list, write_list):
            def find_read_write(op):
                if isinstance(op, tvm.stmt.Store):
                    write_list.append(op)
                if isinstance(op, tvm.expr.Load):
                    read_list.append(op)
            return find_read_write

        stmt = stores[0]
        reads, writes = [], []
        tvm.ir_pass.PostOrderVisit(stmt, make_find_read_write(reads, writes))
        return reads, writes


    @lru_cache(maxsize=None)
    def _get_stores(self, pass_init=True):

        """
        stmt = self.get_loop()
        body = stmt.body #allocate
        body = stmt.body.body #block
        first = body.first #ProducerConsumer, producer of Apad
        rest = body.rest #ProducerConsumer, consumer of Apad

        body = rest.body #ProducerConsumer, producer of O
        print(type(body.func))
        print(body.is_producer)
        print(body.body)
        """

        #TODO: is op_name necessary?
        def make_find_store(op_name, return_list, pass_init=True):
            def find_store(op):
                if isinstance(op, tvm.stmt.Store):
                    if op.buffer_var.name == op_name:
                        value = op.value
                        if pass_init and isinstance(value, tvm.expr.FloatImm) and value.value == 0:
                            return
                        return_list.append(op)
            return find_store

        stores = []
        tvm.ir_pass.PostOrderVisit(self.loop, make_find_store("O", stores, pass_init=pass_init))
        return stores


    @lru_cache(maxsize=None)
    def _get_index_vars(self, op):
        index_vars = []
        index_expr = op.index
        def find_index_vars(op):
            if isinstance(op, tvm.expr.Var):
                index_vars.append(op.name.title())
        tvm.ir_pass.PostOrderVisit(index_expr, find_index_vars)
        return index_vars


    @lru_cache(maxsize=None)
    def computeEnergyCost_UsefulOps(self):
        total_loop_operations = self.get_TripCounts_all_IVs("loop")
        return total_loop_operations * self.get_energy("MAC")


    def computeExecTime_UsefulOps_One_RF_Pass(self):
        total_loop_iterations = self.get_TripCounts_all_IVs("RF")
        if self.env.pe_pipelined:
            return total_loop_iterations
            # return self.env.pe_pipeline_stages + total_loop_iterations - 1
        else:
            #to be implemented later
            return -1


    @lru_cache(maxsize=None)
    def calculate_Cycles_DataDistribution_NOC(self):
        operands = self._get_operands()
        ret_dict = {}
        for operand_name, ops in operands.items():
            op = ops[0]
            assert isinstance(op, tvm.expr.Load)
            #index_expr = operand["stmt"].index
            """
            n, m, c, ox, oy, fx, fy = [ self.get_TripCounts(index, "RF") * self.get_TripCounts(index, "Spatial") for index in self._default_loop_order]
            idx_list = [n, m, c, ox, oy, fx, fy]
            """
            idx_list = [ self.get_TripCounts(index, "RF") * self.get_TripCounts(index, "Spatial") for index in self._default_loop_order]
            communication_cycle = self._get_num_different_pixels(op, idx_list)
            ret_dict[operand_name] = communication_cycle
        return ret_dict


    @lru_cache(maxsize=None)
    def _get_operands(self):
        operands = {}
        def find_ops(op):
            if isinstance(op, tvm.expr.Load) or isinstance(op, tvm.stmt.Store):
                buffer_var_name = op.buffer_var.name
                if buffer_var_name in operands:
                    operands[buffer_var_name].append(op)
                else:
                    operands[buffer_var_name] = [op]
                """
                operands.append({
                    "stmt": op,
                    "op": op.buffer_var
                })
                """
        for stmt in self._get_stores():
            tvm.ir_pass.PostOrderVisit(stmt, find_ops)
        return operands


    @lru_cache(maxsize=None)
    def determine_data_reuse(self, option, user_ordering=None):
        #excluding two innermost levels
        assert(option in self.tiling_levels[:-2])

        data_reuse_list = []
        found_reuse_vectors = []
        for ordering in self._loop_orderings:
            data_reuse_list.append(tuple([ iv+"_"+option for iv in ordering ]))
        ret_dict_with_all_orderings = {}

        if user_ordering != None:
            data_reuse_list.append(tuple([ iv+"_"+option for iv in user_ordering ]))

        operands = self._get_operands()
        order_number = 0
        for ordering in data_reuse_list:
            ret_dict = {}
            factor_additional_comm_read_write_op = {}

            for operand_name, ops in operands.items():
                #TODO: currently only work for the first op
                op = ops[0]
                index_vars = self._get_index_vars(op)
                data_reuse_factor = 1
                operand_minimum_use = 1
                for s in index_vars:
                    assert(s in self.base_TCs)
                    operand_minimum_use *= self.get_TripCounts(s, option)

                # to determine data resue for an operand, we need to traverse
                # ordering for IVs from innermost loop to outer loops
                #for iv in reversed(self._loop_IVs[option]):
                for iv in reversed(list(ordering)):
                    iv = iv.split("_")[0]
                    tc = self.get_TripCounts(iv, option)
                    if tc == 1:
                        continue
                    elif iv not in index_vars:
                        data_reuse_factor *= tc
                    else:
                        break
                ret_dict[operand_name] = data_reuse_factor

                # if read+write operand, we need additional factor
                # for possibly communicating operand data through read network
                reads, writes = self._get_reads_writes_of_operand(operand_name)
                if len(reads) > 0 and len(writes) > 0:
                    total_pass = self.get_TripCounts_all_IVs(option)
                    factor_additional_comm = total_pass / operand_minimum_use
                    factor_additional_comm /= data_reuse_factor
                    if option == self.tiling_levels[0]:
                        factor_additional_comm -= 1
                    if data_reuse_factor == total_pass:
                        factor_additional_comm = 0
                    factor_additional_comm *= operand_minimum_use
                else:
                    factor_additional_comm = 0

                factor_additional_comm_read_write_op[operand_name] = factor_additional_comm

            reuse_vector = []
            for operand_name, ops in operands.items():
                reuse_vector.append(ret_dict[operand_name])
            reuse_vector = tuple(reuse_vector)
            if reuse_vector not in found_reuse_vectors:
                #ret_dict_with_all_orderings[order_number] = (ret_dict, factor_additional_comm_read_write_op)
                ret_dict_with_all_orderings[ordering] = (ret_dict, factor_additional_comm_read_write_op)
                found_reuse_vectors.append(reuse_vector)

            if user_ordering != None and ordering == tuple([iv+"_"+option for iv in user_ordering]):
                return (ret_dict, factor_additional_comm_read_write_op)

            order_number += 1

        return ret_dict_with_all_orderings


    @lru_cache(maxsize=None)
    def _get_reads_writes_of_operand(self, operand_name):
        reads, writes = self._get_reads_writes()
        op_reads, op_writes = [], []
        for read in reads:
            if read.buffer_var.name == operand_name:
                op_reads.append(read)
        for write in writes:
            if write.buffer_var.name == operand_name:
                op_writes.append(write)
        return op_reads, op_writes


    @lru_cache(maxsize=None)
    def calculate_Cycles_InterPECommunication(self):
        reduction_cycles = {}
        reads, writes = self._get_reads_writes()
        read_operands = [ op.buffer_var.name for op in reads ]
        write_operands = [ op.buffer_var.name for op in writes ]
        operands = self._get_operands()
        for operand_name, ops in operands.items():
            if operand_name not in read_operands or \
            operand_name not in write_operands:
                reduction_cycles[operand_name] = 0
                continue
            op = ops[0]
            idx_list = [self.get_TripCounts(index, "RF") for index in self._default_loop_order]
            reduction_cycles_op = self._get_num_different_pixels(
                op, idx_list)
            groups_for_reduction = 1
            reduction_within_a_group = 1
            index_vars = self._get_index_vars(op)
            for iv in self._default_loop_order:
                tc = self.get_TripCounts(iv, "Spatial")
                if iv not in index_vars:
                    reduction_within_a_group *= tc
                else:
                    groups_for_reduction *= tc
            reduction_within_a_group -= 1
            reduction_cycles[operand_name] = \
                reduction_cycles_op * reduction_within_a_group * groups_for_reduction
        return reduction_cycles


    @lru_cache(maxsize=None)
    def calculate_Energy_InterPECommunication(self, breakdown=False):
        energy_costs = {}
        reduction_cycles = self.calculate_Cycles_InterPECommunication()
        unit_energy_cost = 3*self.get_energy("RF") + self.get_energy("network") + self.get_energy("MAC")
        #unit_energy_cost = 0*self.get_energy("RF") + self.get_energy("network") + self.get_energy("MAC")
        for operand_name, cycles in reduction_cycles.items():
            if breakdown:
                ret_tuple = cycles*3*self.get_energy("RF"), cycles*self.get_energy("network"), cycles*self.get_energy("MAC")
                energy_costs[operand_name] = ret_tuple
            else:
                energy_costs[operand_name] = cycles * unit_energy_cost
        return energy_costs


    @lru_cache(maxsize=None)
    def map_operands_to_NOCs(self):
        '''
        allocate interconnect for data communication of operands.
        take into account the interconnect configuration and the
        data reuse of the operands
        Current assumptions (and limitations):
        (a) enough networks are available to communicate all the
            operands simultaneously.
        (b) all networks have same bit-width for the data packets (i.e.
            raw data and configuration processed by multicast controllers).
        (c) each network can communicate 1 data element (of 16-bits)
            during 1 cycle.
        (d) bit-width for row-id and col-id for any data packet is
            sufficiently large i.e. 8-bits.
        TODO_Later: Make it generic for handling x tensor operands via y global
            multicast single-cycle networks.
        TODO_Later: introduce variability in assumptions (b)--(d).
        TODO_Later: Make the routine to handle both interconnect styles
            i.e. systolic (or mesh)  interconnect as well as the
            configuration of current NOC based interconnect mechanism.
            compute total cycles for communication via NOCs
            any operand requires data communication for either of:
            (i)   at every RF pass.
            (ii)  after some RF passes during processing of an SPM pass.
            (iii) after processing all the data for obtained operand elements
                through several SPM passes.
                set values to global array/variables
        '''
        total_RF_passes = self.get_TripCounts_all_IVs("SPM")
        #data_use_factor_spm, _ = self.determine_data_reuse("SPM")
        #data_use_factor_dram, _ = self.determine_data_reuse("DRAM")
        cycles_dataDistribution = self.calculate_Cycles_DataDistribution_NOC()
        cycles_reduction = self.calculate_Cycles_InterPECommunication()

        stall_cycles_useful_ops = self.computeExecTime_UsefulOps_One_RF_Pass()

        ret_dict = {}
        spm_data_use_factors = self.determine_data_reuse("SPM")
        dram_data_use_factors = self.determine_data_reuse("DRAM")

        for spm_ordering, spm_reuse_factor_dict in spm_data_use_factors.items():
            data_use_factor_spm = spm_reuse_factor_dict[0]
            for dram_ordering, dram_reuse_factor_dict in dram_data_use_factors.items():
                data_use_factor_dram = dram_reuse_factor_dict[0]
                ordering = (dram_ordering, spm_ordering)

                # for operand number i, if comm_cycles[i] == x, at every ith RF pass,
                # x cycles will be required for communication via interconnect
                comm_cycles = {}
                skip_comm_op_for_SPM_passes = {}
                for operand_name in self._get_operands():
                    if data_use_factor_spm[operand_name] == total_RF_passes and \
                    data_use_factor_dram[operand_name] > 1:
                        #data will be reused also in DRAM pass
                        skip_comm_op_for_SPM_passes[operand_name] = data_use_factor_dram[operand_name]
                        continue
                    every = data_use_factor_spm[operand_name]
                    comm_cycle = comm_cycles[every] if every in comm_cycles else 0
                    new_comm_cycle = cycles_dataDistribution[operand_name]
                    #reads, writes = self._get_reads_writes_of_operand(operand_name)
                    comm_cycles[every] = max(comm_cycle, new_comm_cycle)
                    skip_comm_op_for_SPM_passes[operand_name] = 1

                stall_cycles = {}
                for every in comm_cycles:
                    stall_cycles[every] = 0

                for operand_name in self._get_operands():
                    if skip_comm_op_for_SPM_passes[operand_name] > 1:
                        #data will be reused also in DRAM pass
                        continue
                    every = data_use_factor_spm[operand_name]
                    reads, writes = self._get_reads_writes_of_operand(operand_name)
                    if len(reads) > 0 and len(writes) > 0:
                        # operand is read-write
                        # assumption: read and write communication for operand done
                        # through separate networks. So no need to consider additional
                        # cycles for cycles_dataDistribution[operand_name].
                        stall_cycles[every] += stall_cycles_useful_ops + cycles_reduction[operand_name]

                ret_dict[ordering] = (comm_cycles, stall_cycles, skip_comm_op_for_SPM_passes)
                #return (comm_cycles, stall_cycles, skip_comm_op_for_SPM_passes)
        return ret_dict


    @lru_cache(maxsize=None)
    def get_Cycles_One_SPM_Pass(self):

        computation_cycles = self.computeExecTime_UsefulOps_One_RF_Pass()
        total_RF_passes = self.get_TripCounts_all_IVs("SPM")

        ret_dict = {}

        cycles_of_all_orderings = self.map_operands_to_NOCs()
        for ordering, reduction_cycle_tuple in cycles_of_all_orderings.items():
            dict_comm_cycles, stall_cycles, _ = reduction_cycle_tuple
            communication_cycles = computation_cycles
            cycles_SPM_pass = computation_cycles * total_RF_passes
            # print(dict_comm_cycles, stall_cycles)
            # Cycles for useful ops included in stall cycles at each RF pass
            # hack to avoid considering cycles for useful ops twice.
            if stall_cycles[1] > 0:
               communication_cycles = 0
               cycles_SPM_pass = 0
            RF_passes_per_operand_use = sorted(list(dict_comm_cycles.keys()))

            for i in RF_passes_per_operand_use:
                if dict_comm_cycles[i] > communication_cycles:
                    cycles_SPM_pass += int((dict_comm_cycles[i] - communication_cycles) * (total_RF_passes / i))
                    communication_cycles = dict_comm_cycles[i]
                cycles_SPM_pass += int(stall_cycles[i] * (total_RF_passes/i))
            ret_dict[ordering] = cycles_SPM_pass
            # print(cycles_SPM_pass)

        return ret_dict

        '''
        total_cycles = 0

        def get_divisors(n):
            for i in range(1, int(math.sqrt(n)+1)):
                if n%i == 0:
                    yield i
            yield n

        disabling this since it is computationally inefficient
        e.g. for 512 RF passes, it will operate on all of them.
        for i in range(1, total_RF_passes+1):
            #calculate cycles required for each RF pass
            divisors = list(get_divisors(i))
            comm_cycle_list = [comm_cycles[divisor] for divisor in divisors if divisor in comm_cycles]
            comm_cycle = max(comm_cycle_list)
            total_cycles += max(computation_cycles, comm_cycle)

        return total_cycles
        '''


    @lru_cache(maxsize=None)
    def determine_DMA_Access(self):
        #keys = ["n", "m", "c", "ox", "oy", "fx", "fy"]
        keys = [ key.lower() for key in self._default_loop_order ]
        values = [ int(self.get_TripCounts(key.title(), "loop") / self.get_TripCounts(key.title(), "DRAM")) for key in self.base_TCs ]
        local_vars = dict(zip(keys, values))
        ret_dict = {}

        for operand_name in self._get_operands():
            reads, writes = self._get_reads_writes_of_operand(operand_name)
            op = (reads + writes)[0]

            input_shape = list(reversed(self._get_tensor_from_name(operand_name).shape))
            num_dimensions = len(input_shape)

            is_continuous_access = True
            dma_count = 1
            data_elements_communicated = 1
            for index in range(num_dimensions):
                max_index_evaluated = int(input_shape[index])
                evaluated = self._get_index_expr_evaluated(op, index, local_vars)
                if not is_continuous_access:
                    dma_count *= evaluated
                else:
                    data_elements_communicated *= evaluated
                if evaluated != max_index_evaluated:
                    is_continuous_access = False
            # Commented out since cretes issue in cmoputing energy cost
            # here we just calculate burst size in terms of total elements
            # data_elements_communicated *= self.env.data_size
            ret_dict[operand_name] = {
                "dma_invocations": dma_count,
                "data_accessed_dma": data_elements_communicated,
            }

        return ret_dict


    def _get_index_expr_evaluated(self, op, index, local_vars_param):
        """
        index: int,
            the index to evaluate (e.g. index=0 returns the evaluation of innermost index expression)
        local_vars: dict,
            keys: variable name, value: the value of variable to evaluate
        """
        index_expr_string = op.index.__repr__()
        tensor = self._get_tensor_from_name(op.buffer_var.name)
        dims = list(reversed(tensor.shape))
        #reduce vaules of `local_vars` by 1 since the expression index starts from 0 (not 1)
        local_vars = local_vars_param.copy()
        for key in local_vars.keys():
            local_vars[key] -= 1
        evaluated = (eval(index_expr_string, {}, local_vars))

        for i in range(index):
            evaluated //= int(dims[i])

        evaluated %= int(dims[index])

        #add +1 to convert 1-base indexing
        return evaluated + 1


    @lru_cache(maxsize=None)
    def calculate_DMA_Cycles(self, burst_size):
        dma_cycles = self.env.dma_init_cycles + self.env.dma_cycles_per_byte * burst_size
        ratio_dataflowAccel_cycles_to_DMA_cycles = self.env.dataflow_accel_frequency / self.env.dma_frequency
        dma_latency = dma_cycles * ratio_dataflowAccel_cycles_to_DMA_cycles
        return math.ceil(dma_latency)


    @lru_cache(maxsize=None)
    def calculate_Cycles_DMA_Access(self):
        dma_cycles = {}
        dma_accesses = self.determine_DMA_Access()
        for operand_name in self._get_operands():
            dma_invocations = dma_accesses[operand_name]["dma_invocations"]
            data_accessed_dma = dma_accesses[operand_name]["data_accessed_dma"] * self.env.data_size
            dma_cycles[operand_name] = dma_invocations * self.calculate_DMA_Cycles(data_accessed_dma)
        return dma_cycles


    @lru_cache(maxsize=None)
    def calculate_Energy_DMA_Access(self):
        dma_access_energy = {}
        dma_accesses = self.determine_DMA_Access()
        for operand_name in self._get_operands():
            dma_invocations = dma_accesses[operand_name]["dma_invocations"]
            data_accessed_dma = dma_accesses[operand_name]["data_accessed_dma"]
            dma_access_energy[operand_name] = dma_invocations * data_accessed_dma * self.get_energy("DRAM")
        return dma_access_energy


    def get_Cycles_One_Layer(self):
        cycles_dataDistribution = self.calculate_Cycles_DataDistribution_NOC()
        #comm_cycles_every_x_RF_pass, _, skip_comm_op_for_SPM_passes = self.map_operands_to_NOCs()
        #cycles_1_RF_pass = comm_cycles_every_x_RF_pass[1]
        #data_use_factor_dram, _ = self.determine_data_reuse("DRAM")
        #data_processing_cycles = self.get_Cycles_One_SPM_Pass()
        dma_cycles = self.calculate_Cycles_DMA_Access()
        cycles_reduction = self.calculate_Cycles_InterPECommunication()
        #dma_comm_cycles = {}
        #comm_cycles = {}
        #op_list = {}

        ret_dict = {}
        reduction_cycles_of_all_orderings = self.map_operands_to_NOCs()
        dram_data_use_factors = self.determine_data_reuse("DRAM")
        for ordering, data_processing_cycles in self.get_Cycles_One_SPM_Pass().items():
            dram_ordering, spm_ordering = ordering
            comm_cycles_every_x_RF_pass, _, skip_comm_op_for_SPM_passes = reduction_cycles_of_all_orderings[ordering]
            cycles_1_RF_pass = comm_cycles_every_x_RF_pass[1]
            data_use_factor_dram, _ = dram_data_use_factors[dram_ordering]
            dma_comm_cycles = {}
            comm_cycles = {}
            op_list = {}
            read_write_op = {}

            for operand_name in self._get_operands():
                every = data_use_factor_dram[operand_name]
                dma_cycle_for_operand = dma_cycles[operand_name]
                reads, writes = self._get_reads_writes_of_operand(operand_name)
                if len(reads) > 0 and len(writes) > 0:
                    read_write_op[operand_name] = True
                    # loads and stores for DMA for adjacent SPM pass are scheduled
                    # simultaneously at every SPM pass
                    if every == 1:
                        dma_cycle_for_operand *= 2
                else:
                    read_write_op[operand_name] = False

                if every in dma_comm_cycles:
                    dma_comm_cycles[every] += dma_cycle_for_operand
                    op_list[every].append(operand_name)
                else:
                    dma_comm_cycles[every] = dma_cycle_for_operand
                    op_list[every] = list(operand_name)

            various_reuse_factors = len(dma_comm_cycles)
            """
            if various_reuse_factors == 1:
                return comm_cycles
            """

            comm_cycles = dict(sorted(dma_comm_cycles.items()))
            keylist = list(comm_cycles.keys())

            # get total dma cycles for every xth pass
            for i in range(various_reuse_factors-1):
                for j in range(i+1,various_reuse_factors):
                    comm_cycles[keylist[j]] += comm_cycles[keylist[i]]

            cycles_1_spm_pass = max(comm_cycles[1], data_processing_cycles)
            total_SPM_pass = self.get_TripCounts_all_IVs("DRAM")
            total_cycles = cycles_1_spm_pass * total_SPM_pass
            for every in comm_cycles:
                if every > 1:
                    for operand_name in op_list[every]:
                        op_use_factor = data_use_factor_dram[operand_name]
                        cycles_1 = 0
                        cycles_2 = 0
                        cycles_pe_array_1 = data_processing_cycles
                        cycles_pe_array_2 = data_processing_cycles
                        dma_cycles_1 = comm_cycles[op_use_factor]
                        dma_cycles_2 = comm_cycles[op_use_factor] - dma_comm_cycles[op_use_factor]

                        if read_write_op[operand_name] == True:
                            dma_cycles_2 += dma_comm_cycles[op_use_factor]

                        if skip_comm_op_for_SPM_passes[operand_name] > 1:
                            # for communication of data between SPM and RF
                            comm_op_NOC = cycles_dataDistribution[operand_name]
                            stall_cycles_comm_between_SPM_RF = abs(max(cycles_1_RF_pass, comm_op_NOC) - cycles_1_RF_pass)
                            cycles_pe_array_2 += stall_cycles_comm_between_SPM_RF

                            if read_write_op[operand_name] == True:
                                #reduction if read+write
                                cycles_pe_array_1 += cycles_reduction[operand_name] + self.computeExecTime_UsefulOps_One_RF_Pass()
                                # need to store data back so that in next spm pass, we can write it back
                                # but this is accounted when we add stall cycles for cycles_pe_array_2
                                # cycles_pe_array_1 += stall_cycles_comm_between_SPM_RF

                        # add overhead between dma and processing cycles
                        cycles_1 = max(dma_cycles_1, cycles_pe_array_1)
                        cycles_2 = max(dma_cycles_2, cycles_pe_array_2)
                        # print(dma_cycles_1)
                        # print(cycles_pe_array_1)
                        # print(dma_cycles_2)
                        # print(cycles_pe_array_2)
                        # print("Cycles")
                        # print(cycles_1)
                        # print(cycles_2)
                        overhead_cycles_1 = abs(max(cycles_1, cycles_1_spm_pass) - cycles_1_spm_pass)
                        # TODO: For overhead_cycles_2, consider cycles_1_RF_pass instead of cycles_1_spm_pass
                        # TODO: Optimize by doing prefetch/write-back in parallel with execution of SPM passes.
                        overhead_cycles_2 = abs(max(cycles_2, cycles_1_spm_pass) - cycles_1_spm_pass)
                        # print(cycles_1_spm_pass)
                        # print(overhead_cycles_1)
                        # print(overhead_cycles_2)
                        total_cycles += (overhead_cycles_1 + overhead_cycles_2) * (total_SPM_pass / op_use_factor)

            ret_dict[ordering] = total_cycles
        return ret_dict


    def get_data_allocated_RF(self):
        data_RF = {}
        tc_list_RF = [self.get_TripCounts(index, "RF") for index in self._default_loop_order]
        operands = self._get_operands()
        for operand_name, ops in operands.items():
            op = ops[0]
            data_RF[operand_name] = self._get_num_different_pixels(op, tc_list_RF)

        return data_RF


    def get_data_allocated_PE_array(self):
        data_PE_array = {}
        tc_list = [self.get_TripCounts(index, "RF") * self.get_TripCounts(index, "Spatial") for index in self._default_loop_order]
        operands = self._get_operands()
        for operand_name, ops in operands.items():
            op = ops[0]
            data_PE_array[operand_name] = self._get_num_different_pixels(op, tc_list)

        return data_PE_array


    def get_data_allocated_SPM(self):
        data_SPM = {}
        tc_list = [self.get_TripCounts(index, "RF") * self.get_TripCounts(index, "Spatial") * self.get_TripCounts(index, "SPM") for index in self._default_loop_order]
        operands = self._get_operands()
        for operand_name, ops in operands.items():
            op = ops[0]
            data_SPM[operand_name] = self._get_num_different_pixels(op, tc_list)

        return data_SPM


    def set_tiling(self, iv, tiling):
        """
        Set the tiling values for `iv`.

        Inputs
        ------
        `iv`: string
            The name of the induction variable (e.g. "N")
        `tiling`: list
            This list containing the tiling values for each level,
            from lowest to highest (e.g. DRAM to Spatial)
        """
        assert(iv in self.base_TCs)
        assert(len(tiling) == len(self.tiling_levels))
        assert(reduce(lambda x, y: x*y, tiling) == self.base_TCs[iv])
        for i, val in enumerate(tiling):
            target_iv = iv + "_" + self.tiling_levels[i]
            self._loop_TCs[target_iv] = val
        self._clear_cache()


    def set_ordering(self, level, new_order):
        assert level in self.tiling_levels
        for iv in new_order:
            assert iv in self._default_loop_order
        self._loop_IVs[level] = [ x+"_"+level for x in new_order ]


    @lru_cache(maxsize=None)
    def get_Energy_DataDistribution_NOC(self):
        '''
        Energy consumption for communicating data to/from PEs via NOC
        does not include inter-PE communication (e.g.for reduction)
        '''
        operands = self._get_operands()
        communication_energy_NOC = {}
        total_PEs = self.get_TripCounts_all_IVs("Spatial")

        tc_list_RF = [self.get_TripCounts(index, "RF") for index in self._default_loop_order]

        for operand_name, ops in operands.items():
            op = ops[0]

            read_op, write_op = self._get_reads_writes_of_operand(operand_name)
            spatial_IV_value = 1
            # for output data, only some PEs unrolled in space communicate
            if write_op:
                # get list of IVs on which this operand is dependent on.
                # for example, ofmap in convolution is dependent on
                # 'n', 'm', 'ox' and 'oy'
                index_vars = self._get_index_vars(op)
                for s in index_vars:
                    assert(s in self.base_TCs)
                    spatial_IV_value *= self.get_TripCounts(s, "Spatial")
            else:
                # all PEs (i.e. all loops with IVs unrolled in space)
                # requires to read the data
                spatial_IV_value = total_PEs

            communication_energy_NOC[operand_name] = self._get_num_different_pixels(op, tc_list_RF) * spatial_IV_value * self.get_energy("network")

        return communication_energy_NOC


    @lru_cache(maxsize=None)
    def get_Energy_DataCommunication_SPM(self):
        '''
        Energy consumption for communicating data between
        SPM and processor (accelerator controller).
        Given data allocation in SPM and RFs of PEs, it computes
        energy required to communicate data once from SPM to RFs.
        Several such iterations can be necessary to use all the data
        hat can be accommodated in SPM.
        '''
        operands = self._get_operands()
        communication_energy_SPM = {}

        tc_list = [self.get_TripCounts(index, "RF") * self.get_TripCounts(index, "Spatial") for index in self._default_loop_order]

        for operand_name, ops in operands.items():
            op = ops[0]
            communication_energy_SPM[operand_name] = self._get_num_different_pixels(op, tc_list) * self.get_energy("SPM")

        return communication_energy_SPM


    @lru_cache(maxsize=None)
    def get_Energy_One_SPM_Pass(self, breakdown=False):
        '''
        for data allocation in SPM, computes total energy consumption
        required for communicating and processing the data
        Assumption: map_operands_to_NOCs( get_loop_ordering(‘SPM’) )
        is already called.
        '''

        total_RF_passes = self.get_TripCounts_all_IVs("SPM")
        #data_use_factor_spm, _ = self.determine_data_reuse("SPM")
        #dict_comm_cycles, _, skip_comm_op_for_SPM_passes = self.map_operands_to_NOCs()
        #RF_passes_per_operand_use = list(dict_comm_cycles.keys())
        communication_energy_NOC = self.get_Energy_DataDistribution_NOC()
        communication_energy_SPM = self.get_Energy_DataCommunication_SPM()
        energy_reduction_NOC = self.calculate_Energy_InterPECommunication(breakdown)
        spm_data_use_factors = self.determine_data_reuse("SPM")
        ret_dict = {}
        ret_dict_breakdown = {}

        for ordering, comm_cycle_tuple in self.map_operands_to_NOCs().items():
            dram_ordering, spm_ordering = ordering
            _, _, skip_comm_op_for_SPM_passes = comm_cycle_tuple
            data_use_factor_spm, _ = spm_data_use_factors[spm_ordering]
            energy_SPM_pass = 0

            energy_breakdown_MAC = 0
            energy_breakdown_RF = 0
            energy_breakdown_NOC = 0
            energy_breakdown_SPM = 0


            for operand_name in self._get_operands():
                energy_op = 0

                energy_op_MAC = 0
                energy_op_RF = 0
                energy_op_NOC = 0
                energy_op_SPM = 0

                if skip_comm_op_for_SPM_passes[operand_name] <= 1:
                    energy_op = communication_energy_NOC[operand_name] + communication_energy_SPM[operand_name]

                    energy_op_NOC = communication_energy_NOC[operand_name]
                    energy_op_SPM = communication_energy_SPM[operand_name]

                    reads, writes = self._get_reads_writes_of_operand(operand_name)
                    if len(reads) > 0 and len(writes) > 0:
                        #operand is read-write
                        # energy spent on reading and writing the data for operand
                        energy_op = 2*energy_op

                        energy_op_NOC *= 2
                        energy_op_SPM *= 2

                        # add energy for reduction performed on PE array
                        if breakdown:
                            energy_op_RF += energy_reduction_NOC[operand_name][0]
                            energy_op_NOC += energy_reduction_NOC[operand_name][1]
                            energy_op_MAC += energy_reduction_NOC[operand_name][2]

                            energy_op += sum(energy_reduction_NOC[operand_name])
                        else:
                            energy_op += energy_reduction_NOC[operand_name]

                        #print(communication_energy_NOC[operand_name], energy_reduction_NOC[operand_name])

                #print(data_use_factor_spm, operand_name, energy_op * (total_RF_passes / data_use_factor_spm[operand_name]))
                energy_SPM_pass += energy_op * (total_RF_passes / data_use_factor_spm[operand_name])

                times = total_RF_passes / data_use_factor_spm[operand_name]
                energy_breakdown_NOC += energy_op_NOC * times
                energy_breakdown_SPM += energy_op_SPM * times
                energy_breakdown_MAC += energy_op_MAC * times
                energy_breakdown_RF += energy_op_RF * times

            ret_dict[ordering] = energy_SPM_pass
            if breakdown:
                #print(energy_SPM_pass, energy_breakdown_MAC+energy_breakdown_RF+energy_breakdown_NOC+energy_breakdown_SPM)
                assert abs(energy_SPM_pass - (energy_breakdown_MAC + energy_breakdown_RF + energy_breakdown_NOC + energy_breakdown_SPM)) < 1.0e-3
            ret_dict_breakdown[ordering] = energy_breakdown_MAC, energy_breakdown_RF, energy_breakdown_SPM, energy_breakdown_NOC

        if breakdown:
            return ret_dict_breakdown
        else:
            return ret_dict


    @lru_cache(maxsize=None)
    def get_Energy_DataCommunication_DRAM(self, breakdown=False):
        #_, _, skip_comm_op_for_SPM_passes = self.map_operands_to_NOCs()
        total_SPM_passes = self.get_TripCounts_all_IVs("DRAM")
        #data_use_factor_dram, factor_additional_comm_read_write_op = self.determine_data_reuse("DRAM")
        dma_access_energy = self.calculate_Energy_DMA_Access()
        communication_energy_NOC = self.get_Energy_DataDistribution_NOC()
        communication_energy_SPM = self.get_Energy_DataCommunication_SPM()
        energy_reduction_NOC = self.calculate_Energy_InterPECommunication()
        energy_reduction_NOC_breakdown = self.calculate_Energy_InterPECommunication(True)

        dram_data_use_factors = self.determine_data_reuse("DRAM")
        ret_dict = {}
        ret_dict_breakdown = {}

        for ordering, reduction_cycle_tuple in self.map_operands_to_NOCs().items():
            dram_ordering, spm_ordering = ordering
            _, _, skip_comm_op_for_SPM_passes = reduction_cycle_tuple
            data_use_factor_dram, factor_additional_comm_read_write_op = dram_data_use_factors[dram_ordering]
            energy_DRAM = 0

            energy_breakdown_MAC = 0
            energy_breakdown_RF = 0
            energy_breakdown_SPM = 0
            energy_breakdown_NOC = 0
            energy_breakdown_DRAM = 0

            for operand_name in self._get_operands():
                comm_energy_DRAM = dma_access_energy[operand_name] * (total_SPM_passes / data_use_factor_dram[operand_name])

                # if read+write operand, we need may need to communicate
                # operand data through read network.
                reads, writes = self._get_reads_writes_of_operand(operand_name)
                if len(reads) > 0 and len(writes) > 0:
                    comm_energy_DRAM += (dma_access_energy[operand_name] * factor_additional_comm_read_write_op[operand_name])

                energy_breakdown_DRAM += comm_energy_DRAM

                if skip_comm_op_for_SPM_passes[operand_name] > 1:
                    energy_op = communication_energy_NOC[operand_name] + communication_energy_SPM[operand_name] + energy_reduction_NOC[operand_name]
                    comm_energy_DRAM += energy_op * (total_SPM_passes / skip_comm_op_for_SPM_passes[operand_name])
                    #print(operand_name, data_use_factor_dram, factor_additional_comm_read_write_op, dma_access_energy[operand_name], energy_reduction_NOC[operand_name], comm_energy_DRAM)

                    times = total_SPM_passes / skip_comm_op_for_SPM_passes[operand_name]
                    energy_breakdown_RF += energy_reduction_NOC_breakdown[operand_name][0] * times
                    energy_breakdown_MAC += energy_reduction_NOC_breakdown[operand_name][2] * times
                    energy_breakdown_SPM += communication_energy_SPM[operand_name] * times
                    energy_breakdown_NOC += (communication_energy_NOC[operand_name] + energy_reduction_NOC_breakdown[operand_name][1]) * times

                energy_DRAM += comm_energy_DRAM
            ret_dict[ordering] = energy_DRAM
            #print(energy_DRAM, (energy_breakdown_DRAM + energy_breakdown_SPM + energy_breakdown_NOC + energy_breakdown_RF + energy_breakdown_MAC))
            #assert abs(energy_DRAM - (energy_breakdown_DRAM + energy_breakdown_SPM + energy_breakdown_NOC + energy_breakdown_RF + energy_breakdown_MAC)) < 1.0e-5
            ret_dict_breakdown[ordering] = energy_breakdown_DRAM, energy_breakdown_SPM, energy_breakdown_NOC, energy_breakdown_RF, energy_breakdown_MAC

        if breakdown:
            return ret_dict_breakdown
        else:
            return ret_dict


    def get_Energy_One_Layer(self):
        energy_ops = self.computeEnergyCost_UsefulOps()
        energy_RF = self.computeEnergyCost_RF()

        total_SPM_pass = self.get_TripCounts_all_IVs("DRAM")
        #energy_SPM_pass = self.get_Energy_One_SPM_Pass() * total_SPM_pass
        #energy_DRAM = self.get_Energy_DataCommunication_DRAM()
        #total_energy = energy_ops + energy_RF + energy_SPM_pass + energy_DRAM
        ret_dict = {}

        dram_datacommunication_energies = self.get_Energy_DataCommunication_DRAM()
        for ordering, one_spm_pass_energy in self.get_Energy_One_SPM_Pass().items():
            energy_SPM_pass = one_spm_pass_energy * total_SPM_pass
            energy_DRAM = dram_datacommunication_energies[ordering]
            total_energy = energy_ops + energy_RF + energy_SPM_pass + energy_DRAM
            ret_dict[ordering] = round(total_energy)

        return ret_dict


    def get_Energy_Distribution(self):
        ret_dict = {}

        #energy_RF = self.computeEnergyCost_RF()
        #energy_ops = self.computeEnergyCost_UsefulOps()
        total_SPM_pass = self.get_TripCounts_all_IVs("DRAM")
        dram_datacommunication_energies = self.get_Energy_DataCommunication_DRAM(breakdown=True)
        spm_energies = self.get_Energy_One_SPM_Pass(breakdown=True)

        for ordering in self.map_operands_to_NOCs().keys():
            energy_MAC_total = self.computeEnergyCost_UsefulOps()
            energy_RF_total = self.computeEnergyCost_RF()
            energy_NOC_total = 0
            energy_SPM_total = 0
            energy_DRAM_total = 0

            energy_MAC, energy_RF, energy_SPM, energy_NOC = spm_energies[ordering]
            energy_MAC_total += energy_MAC * total_SPM_pass
            energy_RF_total += energy_RF * total_SPM_pass
            energy_SPM_total += energy_SPM * total_SPM_pass
            energy_NOC_total += energy_NOC * total_SPM_pass

            energy_DRAM, energy_SPM, energy_NOC, energy_RF, energy_MAC = dram_datacommunication_energies[ordering]
            energy_DRAM_total += energy_DRAM
            energy_SPM_total += energy_SPM
            energy_NOC_total += energy_NOC
            energy_RF_total += energy_RF
            energy_MAC_total += energy_MAC

            ret_dict[ordering] = energy_MAC_total, energy_RF_total, energy_NOC_total, energy_SPM_total, energy_DRAM_total
        return ret_dict


    def generate_loop_orderings(self):
        '''
        orderings are to be used for finding data reuse while communicating to
        lower memories i.e. SPM and DRAM.
        We find all unique orderings that have unique data movement cost and
        return an ordering vector
        The orderings are assumed from outer to inner levels.
        '''
        #excluding two innermost levels
        # assert(option in self.tiling_levels[:-2])
        temp_list = []
        ret_list = []
        operands = self._get_operands()
        for operand_name, ops in operands.items():
            #TODO: currently only work for the first op
            op = ops[0]
            index_vars = self._get_index_vars(op)
            for s in index_vars:
                assert(s in self.base_TCs)

            index_vars_for_op_invariant = []

            for iv in reversed(self._default_loop_order):
                iv = iv.split("_")[0]
                if iv not in index_vars:
                    index_vars_for_op_invariant.append(iv)

            if not index_vars_for_op_invariant:
                continue

            # print(operand_name)
            total_index_vars_for_op_invariant = len(index_vars_for_op_invariant)

            for i in range(total_index_vars_for_op_invariant-1):
                index_vars_for_op_invariant.append("")

            orderings_inner_loops = []
            combination_iv_orderings = list(combinations(index_vars_for_op_invariant, total_index_vars_for_op_invariant))
            total_orderings_op = len(combination_iv_orderings)
            for i in range(total_orderings_op):
                orderings_inner_loops.append(list(filter(lambda x: x!= "", list(combination_iv_orderings[i]))))

            # remove duplicate entries
            new_ordering_list = []
            [new_ordering_list.append(i) for i in orderings_inner_loops if i not in new_ordering_list]

            list_truncating_IVs = []
            total_orderings_op = len(new_ordering_list)
            for i in new_ordering_list:
                truncating_IVs = (list(set(index_vars_for_op_invariant) - set(i)))
                if (truncating_IVs is None) or (total_orderings_op == 1):
                    list_truncating_IVs.append([])
                else:
                    list_truncating_IVs.append(truncating_IVs)

            new_list_truncating_IVs = []
            for i in range(total_orderings_op):
                new_list_truncating_IVs.append(list(filter(lambda x: x!= "", list(list_truncating_IVs[i]))))

            for i in range(total_orderings_op):
                combined_IVs_for_ordering_op = []

                for item in new_ordering_list[i]:
                    combined_IVs_for_ordering_op.append(item)

                combined_IVs_for_ordering_op += index_vars

                if len(new_list_truncating_IVs[i]) != 0:
                    combined_IVs_for_ordering_op += new_list_truncating_IVs[i]

                # print(combined_IVs_for_ordering_op)
                temp_list.append(combined_IVs_for_ordering_op)

        temp_list_sorted = sorted(temp_list, key=lambda x: (x[0], x[1], x[2]))

        for item in temp_list_sorted:
            ret_list.append(tuple(reversed(item)))

        return ret_list


    """
    def get_index_from_ordering(self, ordering):
        return self._loop_index_from_orderings[ordering]


    def get_ordering_from_index(self, index):
        return self._loop_orderings[index]
    """


    def get_ordering_from_reuse_factor(self, reuse_factor, level):
        assert level in self.tiling_levels
        for ordering, reuse_dict in self.determine_data_reuse(level).items():
            if reuse_factor == reuse_dict[0]:
                return ordering
        return None


    def get_min_energy(self):
        energy_of_all_orderings = self.get_Energy_One_Layer()
        dram_ordering, spm_ordering = min(energy_of_all_orderings, key=energy_of_all_orderings.get)
        min_energy = energy_of_all_orderings[(dram_ordering, spm_ordering)]
        return min_energy, dram_ordering, spm_ordering


    def get_min_cycle(self):
        cycles_of_all_orderings = self.get_Cycles_One_Layer()
        dram_ordering, spm_ordering= min(cycles_of_all_orderings, key=cycles_of_all_orderings.get)
        min_cycle = cycles_of_all_orderings[(dram_ordering, spm_ordering)]
        return min_cycle, dram_ordering, spm_ordering


    def get_EDP_One_Layer(self):
        energy_of_all_orderings = self.get_Energy_One_Layer()
        cycles_of_all_orderings = self.get_Cycles_One_Layer()
        edp_of_all_orderings = {}

        for key in energy_of_all_orderings.keys():
            edp_of_all_orderings[key] = energy_of_all_orderings[key] * cycles_of_all_orderings[key]

        return edp_of_all_orderings


    def get_min_edp(self):
        edp_of_all_orderings = self.get_EDP_One_Layer()

        dram_ordering, spm_ordering = min(edp_of_all_orderings, key=edp_of_all_orderings.get)
        min_edp = edp_of_all_orderings[(dram_ordering, spm_ordering)]
        return min_edp, dram_ordering, spm_ordering


    def get_min_edp_energy_cycle(self):
        energy_of_all_orderings = self.get_Energy_One_Layer()
        cycles_of_all_orderings = self.get_Cycles_One_Layer()
        edp_of_all_orderings = {}

        for key in energy_of_all_orderings.keys():
            edp_of_all_orderings[key] = energy_of_all_orderings[key] * cycles_of_all_orderings[key]
        dram_ordering_min_edp, spm_ordering_min_edp = min(edp_of_all_orderings, key=edp_of_all_orderings.get)
        min_edp = edp_of_all_orderings[(dram_ordering_min_edp, spm_ordering_min_edp)]

        dram_ordering_min_energy, spm_ordering_min_energy = min(energy_of_all_orderings, key=energy_of_all_orderings.get)
        min_energy = energy_of_all_orderings[(dram_ordering_min_energy, spm_ordering_min_energy)]

        dram_ordering_min_cycle, spm_ordering_min_cycle = min(cycles_of_all_orderings, key=cycles_of_all_orderings.get)
        min_cycle = cycles_of_all_orderings[(dram_ordering_min_cycle, spm_ordering_min_cycle)]
        # return spm_ordering_min_edp, dram_ordering_min_edp, spm_ordering_min_energy, dram_ordering_min_energy
        return min_edp, min_energy, min_cycle


    def get_min_edp_energy(self):
        """
        energy_of_all_orderings = self.get_Energy_One_Layer()
        cycles_of_all_orderings = self.get_Cycles_One_Layer()
        edp_of_all_orderings = {}
        #print(cycles_of_all_orderings, self.get_Cycles_One_SPM_Pass())
        for key in energy_of_all_orderings.keys():
            edp_of_all_orderings[key] = energy_of_all_orderings[key] * cycles_of_all_orderings[key]
        dram_ordering_min_edp, spm_ordering_min_edp = min(edp_of_all_orderings, key=edp_of_all_orderings.get)
        min_edp = edp_of_all_orderings[(dram_ordering_min_edp, spm_ordering_min_edp)]

        dram_ordering_min_energy, spm_ordering_min_energy = min(energy_of_all_orderings, key=energy_of_all_orderings.get)
        min_energy = energy_of_all_orderings[(dram_ordering_min_energy, spm_ordering_min_energy)]
        # return spm_ordering_min_edp, dram_ordering_min_edp, spm_ordering_min_energy, dram_ordering_min_energy
        """
        min_edp, min_energy, min_cycle = self.get_min_edp_energy_cycle()
        return min_edp, min_energy


    def _clear_cache(self):
        #print(self.calculate_Cycles_DataDistribution_NOC .cache_info())
        self.get_TripCounts_all_IVs.cache_clear()
        self.calculate_Cycles_DataDistribution_NOC.cache_clear()
        self.determine_data_reuse.cache_clear()
        self.calculate_Cycles_DMA_Access.cache_clear()
        self.calculate_Cycles_InterPECommunication.cache_clear()
        self.calculate_Energy_DMA_Access.cache_clear()
        self.calculate_Energy_InterPECommunication.cache_clear()
        self.computeEnergyCost_RF.cache_clear()
        self.computeEnergyCost_UsefulOps.cache_clear()
        self.determine_DMA_Access.cache_clear()
        self.get_Cycles_One_SPM_Pass.cache_clear()
        self.get_Energy_DataCommunication_DRAM.cache_clear()
        self.get_Energy_DataCommunication_SPM.cache_clear()
        self.get_Energy_DataDistribution_NOC.cache_clear()
        self.get_Energy_One_SPM_Pass.cache_clear()
        self.map_operands_to_NOCs.cache_clear()



class ConvLayer(LayerBase):
    def __init__(self, env, **kargs):
        super().__init__(env, **kargs)
        self.channels = kargs["channels"]
        self.kernel_size = int(kargs["kernel_size"][1:-1].split(",")[0])
        self.padding = int(kargs["padding"][1:-1].split(",")[0])
        self.strides = int(kargs["strides"][1:-1].split(",")[0])
        self.output_shape = kargs["output_shape"]
        self.input_shape = kargs["input_shape"]
        if "batch_size" in kargs:
            self.batch_size = kargs["batch_size"]
        elif "batch" in kargs:
            self.batch_size = kargs["batch"]
        else:
            self.batch_size = 1

        self._I = None
        self._W = None
        self._O = None
        self.loop = self._get_loop()

        self.base_TCs = {
            "N": self.batch_size,
            "M": self.output_shape[1],
            "C": self.input_shape[1],
            "Ox": self.output_shape[2],
            "Oy": self.output_shape[2],
            "Fx": self.kernel_size,
            "Fy": self.kernel_size,
        }

        # set loop induction variables and their orders
        self._default_loop_order = ["N", "M", "C", "Ox", "Oy", "Fx", "Fy"]
        default_order = self._default_loop_order
        self._loop_IVs = {}
        for level in self.tiling_levels:
            self._loop_IVs[level] = [ x+"_"+level for x in default_order ]

        # set loop trip counts
        self._loop_TCs = {}
        for iv in self.base_TCs:
            # set DRAM as base_TC and others to 1
            tiling = [self.base_TCs[iv]] + [1]*(len(self.tiling_levels)-1)
            self.set_tiling(iv, tiling)

        self._loop_orderings = self.generate_loop_orderings()
        self._loop_index_from_orderings = {}
        for i, ordering in enumerate(self._loop_orderings):
            self._loop_index_from_orderings[ordering] = i

    def __repr__(self):
        return "name: {}, channels: {}, kernel_size: {}, input_shape: {}, output_shape: {}, padding: {}, strides: {}".format(self.name, self.channels, self.kernel_size, self.input_shape, self.output_shape, self.padding, self.strides)


    def _get_loop(self):
        """
        Returns the lowered form of the loop.

        Returns
        -------
        lowered_func: LoweredFunc
        """

        batch = self.batch_size
        in_channel = self.input_shape[1]
        out_channel = self.output_shape[1]
        in_size = self.input_shape[2]
        kernel = self.kernel_size
        pad = self.padding
        stride = self.strides

        A = tvm.placeholder((batch, in_channel, in_size, in_size), name="A")
        W = tvm.placeholder((out_channel, in_channel, kernel, kernel), name="W")
        out_size = math.floor((in_size - kernel + 2*pad) // stride) + 1
        assert(out_size == self.output_shape[2])

        # insert padding to input
        Apad = tvm.compute(
            (batch, in_channel, in_size + 2*pad, in_size + 2*pad),
            lambda n, c, xx, yy: tvm.if_then_else(
                tvm.all(yy >= pad, yy - pad < in_size,
                        xx >= pad, xx - pad < in_size),
                A[n, c, xx-pad, yy-pad], tvm.const(0., "float32")),
            name='I')

        # Create reduction variables
        rc = tvm.reduce_axis((0, in_channel), name='c')
        ry = tvm.reduce_axis((0, kernel), name='fy')
        rx = tvm.reduce_axis((0, kernel), name='fx')
        # Compute the convolution
        O = tvm.compute(
            (batch, out_channel, out_size, out_size),
            lambda n, m, ox, oy: tvm.sum(
                Apad[n, rc, ox * stride + rx, oy * stride + ry] * W[m, rc, rx, ry],
                axis=[rc, rx, ry]),
            name='O')

        self._I = Apad
        self._W = W
        self._O = O

        self._tensors = {
            "I": self._I,
            "W": self._W,
            "O": self._O,
        }

        s = tvm.create_schedule([O.op])
        n, m, ox, oy = O.op.axis
        s[O].reorder(n, m, rc, ox, oy, rx, ry)

        return tvm.lower(s, [A, W, O], simple_mode=True)


    def _get_num_different_pixels(self, op, idx_list):
        """
        `op`: `tvm.expr.Load` or `tvm.stmt.Store`
        `idx_list`: list of integer representing trip counts of each dimension
            the length and order should be same as self._default_loop_order
        """

        assert(len(idx_list) == len(self._default_loop_order))
        #currently assume the input is given in the order of [n, m, c, ox, oy, fx, fy]
        n, m, c, ox, oy, fx, fy = tuple(idx_list)

        #currently the index is hardcoded
        op_name = op.buffer_var.name
        if op_name == "I":
            return n * c * ((ox-1)*self.strides + fx) * ((oy-1)*self.strides + fy)
        elif op_name == "W":
            return m * c * fx * fy
        elif op_name == "O":
            return n * m * ox * oy
        else:
            assert False

        """
        index_expr = op.index
        operand = getattr(self, "_"+op_name)
        output_shape = operand.shape
        print(operand.value_index)
        print(operand.op.output(0))

        for size in reversed(output_shape):
            pass
        """


    """
    def _get_tensor_from_name(self, name):
        return self._tensors[name]
        ret_dict = {
            "I": self._I,
            "W": self._W,
            "O": self._O,
        }
        assert name in ret_dict
        return ret_dict[name]
    """


class DWConvLayer(ConvLayer):
    def __init__(self, env, **kargs):
        super().__init__(env, **kargs)
        self.channels = kargs["channels"]
        self.kernel_size = int(kargs["kernel_size"][1:-1].split(",")[0])
        self.padding = int(kargs["padding"][1:-1].split(",")[0])
        self.strides = int(kargs["strides"][1:-1].split(",")[0])
        self.output_shape = kargs["output_shape"]
        self.input_shape = kargs["input_shape"]
        if "batch_size" in kargs:
            self.batch_size = kargs["batch_size"]
        elif "batch" in kargs:
            self.batch_size = kargs["batch"]
        else:
            self.batch_size = 1

        self.name += "_depthwise"

        self._I = None
        self._W = None
        self._O = None
        self.loop = self._get_loop()

        self.base_TCs = {
            "N": self.batch_size,
            "M": 1,
            "C": self.input_shape[1],
            "Ox": self.output_shape[2],
            "Oy": self.output_shape[2],
            "Fx": self.kernel_size,
            "Fy": self.kernel_size,
        }

        # set loop induction variables and their orders
        self._default_loop_order = ["N", "M", "C", "Ox", "Oy", "Fx", "Fy"]
        default_order = self._default_loop_order
        self._loop_IVs = {}
        for level in self.tiling_levels:
            self._loop_IVs[level] = [ x+"_"+level for x in default_order ]

        # set loop trip counts
        self._loop_TCs = {}
        for iv in self.base_TCs:
            # set DRAM as base_TC and other TCs to 1
            tiling = [self.base_TCs[iv]] + [1]*(len(self.tiling_levels)-1)
            self.set_tiling(iv, tiling)

        self._loop_orderings = self.generate_loop_orderings()
        self._loop_index_from_orderings = {}
        for i, ordering in enumerate(self._loop_orderings):
            self._loop_index_from_orderings[ordering] = i

    def __repr__(self):
        return "name: {}, channels: {}, kernel_size: {}, input_shape: {}, output_shape: {}, padding: {}, strides: {}".format(self.name, self.channels, self.kernel_size, self.input_shape, self.output_shape, self.padding, self.strides)

    def _get_loop(self):
        """
        Returns the lowered form of the loop.

        Returns
        -------
        lowered_func: LoweredFunc
        """

        batch = self.batch_size
        in_channel = self.input_shape[1]
        out_channel = 1
        in_size = self.input_shape[2]
        kernel = self.kernel_size
        pad = self.padding
        stride = self.strides

        A = tvm.placeholder((batch, in_channel, in_size, in_size), name="A")
        W = tvm.placeholder((out_channel, in_channel, kernel, kernel), name="W")
        out_size = math.floor((in_size - kernel + 2*pad) // stride) + 1
        assert(out_size == self.output_shape[2])

        # insert padding to input
        # over = ((in_size + 2*pad) - kernel) % stride
        # width = (in_size + 2*pad) - over
        width = in_size + 2*pad
        Apad = tvm.compute(
            (batch, in_channel, width, width),
            lambda n, c, xx, yy: tvm.if_then_else(
                tvm.all(yy >= pad, yy - pad < in_size,
                        xx >= pad, xx - pad < in_size),
                A[n, c, xx-pad, yy-pad], tvm.const(0., "float32")),
            name='I')

        # Create reduction variables
        ry = tvm.reduce_axis((0, kernel), name='fy')
        rx = tvm.reduce_axis((0, kernel), name='fx')
        # Compute the convolution
        O = tvm.compute(
            (batch, in_channel, out_size, out_size),
            lambda n, c, ox, oy: tvm.sum(
                Apad[n, c, ox * stride + rx, oy * stride + ry] * W[0, c, rx, ry],
                axis=[rx, ry]),
            name='O')

        self._I = Apad
        self._W = W
        self._O = O

        self._tensors = {
            "I": self._I,
            "W": self._W,
            "O": self._O,
        }

        s = tvm.create_schedule([O.op])
        n, c, ox, oy = O.op.axis
        s[O].reorder(n, c, ox, oy, rx, ry)

        return tvm.lower(s, [A, W, O], simple_mode=True)


    def _get_num_different_pixels(self, op, idx_list):
        """
        `op`: `tvm.expr.Load` or `tvm.stmt.Store`
        `idx_list`: list of integer representing trip counts of each dimension
            the length and order should be same as self._default_loop_order
        """

        assert(len(idx_list) == len(self._default_loop_order))
        #currently assume the input is given in the order of [n, m, c, ox, oy, fx, fy]
        n, m, c, ox, oy, fx, fy = tuple(idx_list)

        #currently the index is hardcoded
        op_name = op.buffer_var.name
        if op_name == "I":
            return n * c * ((ox-1)*self.strides + fx) * ((oy-1)*self.strides + fy)
        elif op_name == "W":
            return c * fx * fy
        elif op_name == "O":
            return n * c * ox * oy
        else:
            assert False


class GemmLayer(LayerBase):
    def __init__(self, env, **kargs):
        super().__init__(env, **kargs)

        self.base_TCs = None
        self._loop_TCs = None
        self._loop_IVs = None
        self._default_loop_order = None
        self.loop = None
        self._loop_orderings = None

        M, N, K = kargs["M"], kargs["N"], kargs["K"]

        k = tvm.reduce_axis((0, K), 'k')
        A = tvm.placeholder((M, K), name="A")
        B = tvm.placeholder((K, N), name="B")
        C = tvm.compute(
            (M,N),
            lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),
            name="C"
        )
        s = tvm.create_schedule(C.op)
        self.loop = tvm.lower(s, [A, B, C], simple_mode=True)

        self._A = A
        self._B = B
        self._C = C

        self._tensors = {
            "A": self._A,
            "B": self._B,
            "C": self._C,
        }

        self.base_TCs = {
            "X": M,
            "Y": N,
            "K": K,
        }

        self._default_loop_order = ["X", "Y", "K"]
        default_order = self._default_loop_order
        self._loop_IVs = {}
        for level in self.tiling_levels:
            self._loop_IVs[level] = [ x+"_"+level for x in default_order ]

        # set loop trip counts
        self._loop_TCs = {}
        for iv in self.base_TCs:
            # set DRAM = base_TC and others to 1
            tiling = [self.base_TCs[iv]] + [1]*(len(self.tiling_levels)-1)
            self.set_tiling(iv, tiling)

        self._loop_orderings = self.generate_loop_orderings()
        self._loop_index_from_orderings = {}
        for i, ordering in enumerate(self._loop_orderings):
            self._loop_index_from_orderings[ordering] = i

        tc_list_RF = [self.get_TripCounts(index, "DRAM") for index in self._default_loop_order]
        for operand_name, ops in self._get_operands().items():
            op = ops[0]
            self._get_num_different_pixels(op, tc_list_RF)


    def __repr__(self):
        return "{}: (M, K) * (K, N), M={}, K={}, N={}".format(
            self.name,
            self.base_TCs["X"], self.base_TCs["K"], self.base_TCs["Y"]
        )


    @lru_cache(maxsize=None)
    def _get_stores(self, pass_init=True):
        def make_find_store(return_list, pass_init=True):
            def find_store(op):
                if isinstance(op, tvm.stmt.Store):
                    value = op.value
                    if pass_init and isinstance(value, tvm.expr.FloatImm) and value.value == 0:
                        return
                    return_list.append(op)
            return find_store

        stores = []
        tvm.ir_pass.PostOrderVisit(self.loop, make_find_store(stores, pass_init=pass_init))
        return stores

    def _get_num_different_pixels(self, op, idx_list):
        """
        `op`: `tvm.expr.Load` or `tvm.stmt.Store`
        `idx_list`: list of integer representing trip counts of each dimension
            the length and order should be same as self._default_loop_order
        """

        assert(len(idx_list) == len(self._default_loop_order))
        #currently assume the input is given in the order of [x, y, k]
        x, y, k = tuple(idx_list)

        #currently the index is hardcoded
        op_name = op.buffer_var.name
        if op_name == "A":
            return x * k
        elif op_name == "B":
            return k * y
        elif op_name == "C":
            return x * y
        else:
            assert False




"""
class Environment:
    def __init__(self,
                 pe_pipelined=True,
                 pe_pipeline_stages=1,
                 data_type_size_in_bytes=2,
                 dma_init_cycles=291,
                 dma_cycles_per_byte=0.24,
                 dma_frequency=3200,
                 dataflow_accel_frequency=500,
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
"""


class Dataflow:
    lowered_funcs = []
    enable_analysis = False
    current_func_name = None
    layers = []
    conv_layers = []
    energy_cost = [
        0.96,  # RF access
        2,     # Inter-PE/Multicast Communication to PE
        13.5,  # Accessing SPM (~100 kB)
        200,   # Accessing DRAM
        0.075, # MACs
    ]


def print_node(op):
    if isinstance(op, tvm.stmt.For):
        Dataflow.current_layer.loops.append(op)


def print_node_pass(stmt):
    layer = Layer(len(Dataflow.layers), Dataflow.current_func_name)
    Dataflow.layers.append(layer)
    Dataflow.current_layer = layer
    tvm.ir_pass.PostOrderVisit(stmt, print_node)

    return stmt


def parse_layers(sym):
    attrs = []
    layers = []
    debug_str = sym.debug_str()
    layer = {}
    for line in debug_str.split("\n"):
        line = line.strip()
        if line.startswith("Op"):
            layers.append(layer)
            layer = {}
            op, name = line.split()
            layer["op"] = op.split(":")[1][:-1]
            layer["name"] = name.split("=")[1]

        attrs = ["channels", "kernel_size", "padding", "strides", "units"]
        for attr in attrs:
            if line.startswith(attr):
                layer[attr] = line.split("=")[1]
    layers.append(layer)
    layers = layers[1:]

    return layers


def parse_conv_layers(sym, target, shape_dict, params, env, batch_size):
    """
    Parse the graph `sym` and find conv layers.
    Update `Dataflow.conv_layers` with the found layers.

    Returns
    -------
    conv_layers: list
        list of `ConvLayer`
    """
    # TODO: Supply batch_size from input_shape directly i.e., through
    # downloading models for a specific batch size.

    layers = parse_layers(sym)

    conv_layers = []
    for layer in layers:
        if "op" in layer and "conv" in layer["op"]:
            conv_layers.append(layer)

    with build_module.build_config(opt_level=0):
        Dataflow.enable_analysis = True
        graph, lib, params = build_module.build(sym, target, shape_dict, params=params)
        Dataflow.enable_analysis = False

    shape = graph.json_attr("shape")
    json_graph = json.loads(graph.json())

    nodes = []
    output_shape = {}
    input_shape = {}
    weights_shape = {}
    for i, node in enumerate(json_graph["nodes"]):
        name = node["name"]
        output_shape[name] = shape[i]
        node_info = {
            "index": i,
            "name": name,
            "shape": shape[i],
        }
        try:
            func_name = node["attrs"]["func_name"]
            node_info["func_name"] = func_name
            #find input shape
            input_index = node["inputs"][0][0]
            input_shape[name] = shape[input_index]
            node_info["input_shape"] = shape[input_index]

            #find shape of weights
            weights_index = node["inputs"][1][0]
            weights_shape[name] = shape[weights_index]
            node_info["weights_shape"] = shape[weights_index]
        except:
            pass
        nodes.append(node_info)

    parsed_layers = []
    # for layer in layers:
    #     if "op" in layer and "conv" in layer["op"]:
    #         name = layer["name"]
    #         layer["output_shape"] = output_shape[name]
    #         layer["input_shape"] = input_shape[name]
    #         layer["batch_size"] = batch_size
    #         conv_layer = ConvLayer(env=env, **layer)
    #         parsed_layers.append(conv_layer)
    #     elif "op" in layer and "dense" in layer["op"]:
    #         name = layer["name"]
    #         layer["output_shape"] = output_shape[name]
    #         layer["input_shape"] = input_shape[name]
    #         layer["M"] = int(input_shape[name][0]) * batch_size
    #         layer["N"] = int(layer["units"])
    #         layer["K"] = int(input_shape[name][1])
    #         conv_layer = GemmLayer(env=env, **layer)
    #         parsed_layers.append(conv_layer)
    #     else: #not yet supported layer type
    #         parsed_layers.append(layer)

    for idx, layer in enumerate(layers):
        layer["index"] = idx
        if ("op" in layer) and ("conv2d" in layer["op"]):
            name = layer["name"]
            layer["output_shape"] = output_shape[name]
            layer["input_shape"] = input_shape[name]
            layer["batch_size"] = batch_size
            if weights_shape[name][1] == 1:
                conv_layer = DWConvLayer(env=env, **layer)
            else:
                conv_layer = ConvLayer(env=env, **layer)
            parsed_layers.append(conv_layer)
        elif ("op" in layer) and ("dense" in layer["op"]):
            name = layer["name"]
            layer["output_shape"] = output_shape[name]
            layer["input_shape"] = input_shape[name]
            layer["M"] = int(input_shape[name][0]) * batch_size
            layer["N"] = int(layer["units"])
            layer["K"] = int(input_shape[name][1])
            gemm_layer = GemmLayer(env=env, **layer)
            parsed_layers.append(gemm_layer)
        else: #not yet supported layer type
            parsed_layers.append(layer)

    return parsed_layers


def get_dataflow(sym, target, shape_dict, params, batch_size=1):
    #log.setLevel(logging.DEBUG)
    env = expr_parameters.Environment()
    conv_layers = parse_conv_layers(sym, target, shape_dict, params, env, batch_size)
    return conv_layers
