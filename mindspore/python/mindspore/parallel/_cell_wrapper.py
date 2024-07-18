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

from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
from mindspore.ops.operations.comm_ops import AllGather
from mindspore.communication import GlobalComm
from mindspore.common import jit

_ALLGATHER_CELL = None


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


def _single_parameter_broadcast(net, layout, cur_rank=0, initial_rank=0):
    """
    Broadcast single parameter to other rank in data parallel dimension.
    """
    if not layout:
        return
    import mindspore as ms
    from mindspore import Tensor
    from mindspore.communication import get_rank, create_group, get_group_size
    from mindspore.train._utils import get_parameter_redundancy, remove_param_redundancy

    origin_parallel_mode = ms.get_auto_parallel_context("parallel_mode")
    if origin_parallel_mode not in ("semi_auto_parallel", "auto_parallel"):
        return
    if cur_rank != get_rank():
        raise ValueError(f"For parameter broadcast, the cur_rank: {cur_rank} is wrong.")
    if initial_rank % (get_group_size() / ms.get_auto_parallel_context("pipeline_stages")) != 0:
        raise ValueError(f"For parameter broadcast, the initial_rank: {initial_rank} is wrong.")
    param_redundancy = get_parameter_redundancy(layout, initial_rank)
    if not param_redundancy:
        return
    single_params = remove_param_redundancy(param_redundancy)
    if not single_params:
        return
    param_redundancy_reversed = {}
    for key, redundancy in param_redundancy.items():
        for item in redundancy:
            if len(item) == 1:
                continue
            if cur_rank in item:
                param_redundancy_reversed.setdefault(item, []).append(key)
    if not param_redundancy_reversed:
        return
    if cur_rank not in single_params:
        return
    net_param_dict = net.parameters_dict()
    ms.set_auto_parallel_context(parallel_mode="hybrid_parallel")
    for group, params in param_redundancy_reversed.items():
        create_group(str(group), list(group))
        allreduce_input = []
        for param in params:
            if param not in net_param_dict:
                raise ValueError("For parameter broadcast, the param: {param} can not be found.")
            real_param = net_param_dict[param]
            if param not in single_params[cur_rank]:
                real_param.set_data(Tensor(np.zeros(real_param.shape), dtype=real_param.dtype))
            allreduce_input.append(real_param)
        if not allreduce_input:
            continue
        communicator = SingleCommunicator(str(group))
        for real_param in allreduce_input:
            real_param.set_data(communicator(real_param), real_param.sliced)
    ms.set_auto_parallel_context(parallel_mode=origin_parallel_mode)
