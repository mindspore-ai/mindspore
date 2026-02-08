# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
The tests of mindspore, used to test communication for mint.distributed.
"""
import numpy as np
import pytest
import mindspore as ms
from mindspore import context
from mindspore.graph.api import _pynative_executor
from mindspore.ops.communication import (
    init_process_group,
    get_rank,
    get_world_size,
    gather_into_tensor,
    scatter_tensor,
)

#msrun --worker_num=8 --local_worker_num=8 --master_port=10923 --bind_core True --join True --cluster_time_out=800  pytest -sv --disable-warnings test_distributed.py
np.random.seed(1)
init_process_group()
context.set_auto_parallel_context(
    parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True
)
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
rank = get_rank()
size = get_world_size()
if size % 2 != 0:
    raise RuntimeError("Group size should be divided by 2 exactly.")


def test_hccl_scatter_tensor_error():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([3 * size, 3]).astype(np.float32))
    if rank != 0:
        input_tensor = ms.Tensor(np.zeros([3 * size, 3]).astype(np.float32))
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    # 异常场景
    with pytest.raises(TypeError):
        scatter_tensor(1)
    with pytest.raises(TypeError):
        scatter_tensor(output_tensor, input_tensor, src="test")
    with pytest.raises(TypeError):
        scatter_tensor(output_tensor, input_tensor, group=1)
    with pytest.raises(TypeError):
        scatter_tensor(output_tensor, input_tensor, async_op="test")
    with pytest.raises(TypeError):
        scatter_tensor([1], input_tensor)
    with pytest.raises(TypeError):
        scatter_tensor(output_tensor, [1])
    output_tensor = ms.Tensor(np.zeros([1, 3]).astype(np.float32))
    with pytest.raises(ValueError):
        scatter_tensor(output_tensor, input_tensor, src=rank)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 1]).astype(np.float32))
    with pytest.raises(ValueError):
        scatter_tensor(output_tensor, input_tensor, src=rank)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.int32))
    with pytest.raises(ValueError):
        scatter_tensor(output_tensor, input_tensor, src=rank)
        _pynative_executor.sync()


def test_hccl_gather_into_tensor_error():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_tensor = ms.Tensor(np.zeros([3 * size, 3]).astype(np.float32))
    # 异常场景
    with pytest.raises(TypeError):
        gather_into_tensor(1)
    with pytest.raises(TypeError):
        gather_into_tensor(output_tensor, input_tensor, group=1)
    with pytest.raises(TypeError):
        gather_into_tensor(output_tensor, input_tensor, async_op="test")
    with pytest.raises(TypeError):
        gather_into_tensor([1], input_tensor)
    with pytest.raises(TypeError):
        gather_into_tensor(output_tensor, [1])
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    with pytest.raises(ValueError):
        gather_into_tensor(output_tensor, input_tensor, dst=rank)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 3 * size]).astype(np.float32))
    with pytest.raises(ValueError):
        gather_into_tensor(output_tensor, input_tensor, dst=rank)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3 * size, 3]).astype(np.int32))
    with pytest.raises(ValueError):
        gather_into_tensor(output_tensor, input_tensor, dst=rank)
        _pynative_executor.sync()
