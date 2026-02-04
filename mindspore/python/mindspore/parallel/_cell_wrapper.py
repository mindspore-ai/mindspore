# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Cell of auto parallel"""
from __future__ import absolute_import
from __future__ import division

import numpy as np

import mindspore.log as logger
from mindspore import context
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
from mindspore.ops.operations.comm_ops import AllGather
from mindspore.communication import GlobalComm, get_rank
from mindspore.common import jit
from mindspore.communication import create_group, destroy_group, get_group_size
from mindspore.communication._comm_helper import _get_group_map, _remove_group_info
from mindspore.parallel._utils import _mstx_range_decorator
from mindspore.train._utils import get_parameter_redundancy, remove_param_redundancy
from mindspore.parallel.shard import Layout

_ALLGATHER_CELL = None
ALLREDUCE_GROUP_LIST = []


class AllGatherCell(Cell):
    """
    Allgather cell, used in model parallel scenario.
    To allgather the selected parameter slice from each device.
    """

    def __init__(self, group, do_reshape, after_reshape_slice_shape):
        super(AllGatherCell, self).__init__(auto_prefix=False)
        self.allgather = AllGather(group)
        self.do_reshape = do_reshape
        self.after_reshape_slice_shape = tuple(after_reshape_slice_shape)
        self.add_flags(skip_auto_parallel_compile=True)

    @jit()
    def construct(self, x):
        if self.do_reshape:
            x = P.Reshape()(x, self.after_reshape_slice_shape)
        x = self.allgather(x)
        return x


class SaveOptShardCkptCell(Cell):
    """
    Allgather cell, used in optimizer parallel scenario.
    Firstly gather the tensor to original layout in the specified device group.
    Then gather the whole parameter slices from all devices.

    Note:
        This could be optimized later with less communication consumption.
    """

    def __init__(self, group, do_reshape, after_reshape_slice_shape):
        super(SaveOptShardCkptCell, self).__init__(auto_prefix=False)
        self.allgather1 = AllGather(group)
        self.allgather2 = AllGather()
        self.do_reshape = do_reshape
        self.after_reshape_slice_shape = tuple(after_reshape_slice_shape)
        self.add_flags(skip_auto_parallel_compile=True)

    def construct(self, x):
        x = self.allgather1(x)
        if self.do_reshape:
            x = P.Reshape()(x, self.after_reshape_slice_shape)
        x = self.allgather2(x)

        return x


class SingleCommunicator(Cell):
    """
    Used to broadcast single parameter.
    """

    def __init__(self, group_name):
        super(SingleCommunicator, self).__init__()
        self.allreduce = P.AllReduce(group=group_name)
        self.add_flags(skip_auto_parallel_compile=True)

    def construct(self, loaded_param):
        result = self.allreduce(loaded_param)
        return result


def get_allgather_cell(group, need_merge_twice=False, do_reshape=False, after_reshape_slice_shape=()):
    """Get AllGatherCell object."""
    global _ALLGATHER_CELL
    if need_merge_twice:
        _ALLGATHER_CELL = SaveOptShardCkptCell(group, do_reshape, after_reshape_slice_shape)
    else:
        if group:
            _ALLGATHER_CELL = AllGatherCell(group, do_reshape, after_reshape_slice_shape)
        else:
            _ALLGATHER_CELL = AllGatherCell(GlobalComm.WORLD_COMM_GROUP, do_reshape, after_reshape_slice_shape)
    return _ALLGATHER_CELL


def destroy_allgather_cell():
    """Destroy AllGatherCell object."""
    global _ALLGATHER_CELL
    if _ALLGATHER_CELL:
        _ALLGATHER_CELL = None


def _chang_parallel_context(origin_dataset_strategy):
    """Change the original parallel state."""
    context.set_auto_parallel_context(parallel_mode="hybrid_parallel")
    if origin_dataset_strategy != "data_parallel":
        context.set_auto_parallel_context(dataset_strategy="data_parallel")


def _restore_parallel_context(origin_parallel_mode, origin_dataset_strategy):
    """Restore the original parallel state."""
    context.set_auto_parallel_context(parallel_mode=origin_parallel_mode)
    if origin_dataset_strategy != "data_parallel":
        if origin_dataset_strategy is not None and isinstance(origin_dataset_strategy, list):
            origin_dataset_strategy = tuple(tuple(ds_item) for ds_item in origin_dataset_strategy)
        context.set_auto_parallel_context(dataset_strategy=origin_dataset_strategy)


def _get_group_name(group_map, group):
    """get group name"""
    group_name = "remove_redundancy" + str(group)
    is_manual_communication_group = True
    if group_map:
        for name, rank_list in group_map.items():
            if list(group) == rank_list:
                group_name = name
                is_manual_communication_group = False
                break
    return group_name, is_manual_communication_group


def _get_param_redundancy_reversed(param_redundancy, cur_rank):
    """Generate the reverse mapping of parameter redundancy based on the current rank."""
    param_redundancy_reversed = {}
    for key, redundancy in param_redundancy.items():
        for item in redundancy:
            if len(item) == 1:
                continue
            if cur_rank in item:
                param_redundancy_reversed.setdefault(item, []).append(key)
    return param_redundancy_reversed


def _remove_param_not_load(param_name, param_not_load):
    """Remove param_name from param_not_load."""
    if param_not_load is not None and param_name in param_not_load:
        param_not_load.remove(param_name)


def _get_param_index_in_group(total_param_loaded, group, param):
    """Get param_index in group."""
    param_rank_index = []
    for rank_id in group:
        if rank_id < len(total_param_loaded):
            if param in total_param_loaded[rank_id]:
                param_rank_index.append(rank_id)
        else:
            raise ValueError("rank_id should be smaller than total rank num")
    return param_rank_index


def _communicate_allreduce(allreduce_input, group_map, group):
    """Communicate allreduce input."""
    if not allreduce_input:
        return
    from mindspore import Tensor
    group_name, is_manual_communication_group = _get_group_name(group_map, group)
    if is_manual_communication_group:
        create_group(group_name, list(group))
    communicator = SingleCommunicator(group_name)
    for real_param in allreduce_input:
        real_param.set_data(communicator(Tensor(real_param)), real_param.sliced)
    if is_manual_communication_group:
        destroy_group(group_name)
        _remove_group_info(group_name)


def _create_allreduce_input(params, group, net_param_dict, total_param_loaded, param_not_load, cur_rank):
    """Creates allreduce input."""
    from mindspore import Tensor
    allreduce_input = []
    for param in params:
        if param not in net_param_dict:
            continue
        if param.startswith("accu_grads") or param.endswith("expert_load"):
            continue
        param_rank_index = _get_param_index_in_group(total_param_loaded, group, param)
        if not param_rank_index:
            continue
        elif len(param_rank_index) == 1:
            real_param = net_param_dict[param]
            _remove_param_not_load(real_param.name, param_not_load)
            if cur_rank != param_rank_index[0]:
                real_param.set_data(Tensor(np.zeros(real_param.shape), dtype=real_param.dtype), real_param.sliced)
            allreduce_input.append(real_param)
        elif len(param_rank_index) > 1:
            raise ValueError(f"For param {param} in group {group} should be in one rank, but in {param_rank_index}.")
    return allreduce_input


def _get_sorted_group_map():
    """Get the world group map."""
    group_map = _get_group_map()
    if group_map:
        group_map = {key: group_map[key] for key in sorted(group_map.keys())}
    return group_map


def _check_total_param_loaded(total_param_loaded):
    """Check total_param_loaded."""
    flag = True
    for rank_id, param_loaded in enumerate(total_param_loaded):
        if rank_id not in param_loaded:
            flag = False
            logger.warning("The order of loaded parameters on each card obtained by all_gather_object is incorrect,"
                           "and the parameter broadcast will reorder them.")
            break
    if not flag:
        new_total_param_loaded = [None] * len(total_param_loaded)
        for _, param_loaded in enumerate(total_param_loaded):
            for param in param_loaded:
                if isinstance(param, int):
                    new_total_param_loaded[param] = param_loaded
                    break
        return new_total_param_loaded
    return total_param_loaded


@_mstx_range_decorator("parameter_broadcast", domain="model_preparation")
def _single_parameter_broadcast(net, layout, param_not_load=None, param_loaded=None):
    """
    Broadcast single parameter to other rank in data parallel dimension.
    """
    logger.info("Start loading the parameter broadcast for removing redundant parameters.")
    from mindspore.runtime import synchronize
    from mindspore.mint.distributed import all_gather_object
    origin_parallel_mode = context.get_auto_parallel_context("parallel_mode")
    origin_dataset_strategy = context.get_auto_parallel_context("dataset_strategy")
    cur_rank = get_rank()
    if layout:
        param_redundancy = get_parameter_redundancy(layout)
    else:
        param_redundancy = get_parameter_redundancy(net)
    if not param_redundancy:
        return
    single_params = remove_param_redundancy(param_redundancy)
    if not single_params:
        return
    param_redundancy_reversed = _get_param_redundancy_reversed(param_redundancy, cur_rank)
    if not param_redundancy_reversed:
        return
    net_param_dict = net.parameters_dict()
    _chang_parallel_context(origin_dataset_strategy)
    param_loaded.add(cur_rank)
    total_num = get_group_size()
    total_param_loaded = [None] * total_num
    synchronize()
    all_gather_object(total_param_loaded, param_loaded)
    total_param_loaded = _check_total_param_loaded(total_param_loaded)
    group_map = _get_sorted_group_map()
    for group, params in param_redundancy_reversed.items():
        allreduce_input = _create_allreduce_input(params, group, net_param_dict, total_param_loaded, param_not_load,
                                                  cur_rank)
        _communicate_allreduce(allreduce_input, group_map, group)
    _restore_parallel_context(origin_parallel_mode, origin_dataset_strategy)
    synchronize()
    logger.info("End loading the parameter broadcast for removing redundant parameters.")


def _insert_virtual_pp_dim(layout):
    """insert virtual pp dim in device matrix and create new layout"""
    if len(layout.to_dict()["rank_list"]) == get_group_size():
        return layout
    remain_pp = get_group_size() // len(layout.to_dict()["rank_list"])
    layout_info = layout.to_dict()
    device_matrix = layout_info["device_matrix"]
    tensor_map = layout_info["tensor_map"]
    alias_name = layout_info["alias_name"]
    new_devmat = Layout((remain_pp,) + device_matrix, ("remain_pp",) + alias_name)
    tensor_map_alias_name = []
    for val in tensor_map:
        sub_alias_name = []
        if isinstance(val, tuple):
            for sub_val in val:
                if sub_val == -1:
                    sub_alias_name.append("None")
                else:
                    sub_alias_name.append(alias_name[len(device_matrix) - sub_val - 1])
            tensor_map_alias_name.append(tuple(sub_alias_name))
        else:
            if val == -1:
                tensor_map_alias_name.append("None")
            else:
                tensor_map_alias_name.append(alias_name[len(device_matrix) - val - 1])
    new_layout = new_devmat(*tensor_map_alias_name)
    return new_layout


class CommTensorDataForPP(Cell):
    """Communicate tensor data for pipeline parallel scenario."""

    def __init__(self, src_dtensor_info, dst_dtensor_info):
        super().__init__()
        self.zeros = P.Zeros()

        self._current_rank_id = get_rank()
        self._from_dev_num_in_stage = len(src_dtensor_info.layout.to_dict()["rank_list"])
        self._from_rank_id = src_dtensor_info.layout.to_dict()["rank_list"]
        self._current_rank_has_data = self._current_rank_id in src_dtensor_info.layout.to_dict()["rank_list"]
        self._diff_rank_id = [
            rank_id for rank_id in dst_dtensor_info.layout.to_dict()["rank_list"] if rank_id not in self._from_rank_id]
        self._group, self._root_idx = self._create_all_reduce_group()

    def comm_data(self, comm_data):
        """communicate data"""
        from mindspore import mint
        comm_handle = mint.distributed.broadcast(comm_data, self._root_idx, self._group, async_op=False)
        return comm_handle

    def _create_all_reduce_group(self):
        """create all reduce group"""
        global ALLREDUCE_GROUP_LIST
        current_rank_stage_id = self._current_rank_id // self._from_dev_num_in_stage
        end_stage = self._from_dev_num_in_stage * (current_rank_stage_id + 1)
        start_stage = self._from_dev_num_in_stage * current_rank_stage_id
        rank_pos_in_stage = list(range(start_stage, end_stage)).index(self._current_rank_id)
        root_idx = self._from_rank_id[rank_pos_in_stage]
        all_reduce_rank_list = [self._from_rank_id[rank_pos_in_stage]]
        while rank_pos_in_stage < len(self._diff_rank_id):
            all_reduce_rank_list.append(self._diff_rank_id[rank_pos_in_stage])
            rank_pos_in_stage += self._from_dev_num_in_stage
        all_reduce_rank_list.sort()
        str_rank_list = '-'.join([str(rank) for rank in all_reduce_rank_list])
        all_reduce_group = f"pp_allreduce_group-{str_rank_list}"
        if all_reduce_group in ALLREDUCE_GROUP_LIST:
            return all_reduce_group, root_idx
        ALLREDUCE_GROUP_LIST.append(all_reduce_group)
        create_group(all_reduce_group, all_reduce_rank_list)
        logger.debug(f"Create group {all_reduce_group} for tensor data communication.")
        return all_reduce_group, root_idx


class RedistributionCell(Cell):
    """Redistribute src_layout to dst_layout"""

    def __init__(self, src_layout, dst_layout):
        super().__init__()
        if src_layout is None or dst_layout is None:
            raise ValueError("src_layout and dst_layout should not be None.")
        self._total_dev_num = get_group_size()
        src_layout = _insert_virtual_pp_dim(src_layout)
        dst_layout = _insert_virtual_pp_dim(dst_layout)
        self.src_identity = P.Identity().shard(in_strategy=(src_layout,), out_strategy=(src_layout,))
        self.src_identity.add_prim_attr("self_define_shard", True)
        self.dst_identity = P.Identity().shard(in_strategy=(dst_layout,), out_strategy=(dst_layout,))
        self.dst_identity.add_prim_attr("self_define_shard", True)

    def construct(self, input_tensor):
        """run redistribution"""
        src_tensor = self.src_identity(input_tensor)
        dst_tensor = self.dst_identity(src_tensor)
        return dst_tensor
