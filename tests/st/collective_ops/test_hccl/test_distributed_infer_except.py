# Copyright 2024 Huawei Technologies Co., Ltd
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
The tests of mindspore, used to test infer exceptions for mint.distributed.
"""
import numpy as np
import pytest
import mindspore as ms
from mindspore import context
from mindspore.graph.api import _pynative_executor
from mindspore.mint.distributed.distributed import (
    init_process_group,
    get_rank,
    get_world_size,
    all_gather_into_tensor,
    all_gather_into_tensor_uneven,
    reduce_scatter_tensor_uneven,
    all_to_all,
    all_to_all_single,
    reduce_scatter_tensor,
    gather,
    scatter,
    all_gather,
    reduce_scatter,
)

#msrun --worker_num=8 --local_worker_num=8 --master_port=10923 --bind_core True --join True --cluster_time_out=800  pytest -sv --disable-warnings test_distributed_infer_except.py
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


def log_function_entry_exit(func):
    """
    Feature: log function entry exit
    Description: add log for func
    Expectation: success
    """
    def wrapper(*args, **kwargs):
        # 打印进入函数的信息
        print(f"Entering comm function: {func.__name__}", flush=True)
        # 调用原函数
        result = func(*args, **kwargs)
        # 打印退出函数的信息
        print(f"Exiting comm function: {func.__name__}", flush=True)
        return result
    return wrapper


@log_function_entry_exit
def test_infer_hccl_all_gather_into_tensor():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # infer异常
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    with pytest.raises(ValueError):
        all_gather_into_tensor(output_tensor, input_tensor)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 3 * size]).astype(np.float32))
    with pytest.raises(ValueError):
        all_gather_into_tensor(output_tensor, input_tensor)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3 * size, 3]).astype(np.int32))
    with pytest.raises(ValueError):
        all_gather_into_tensor(output_tensor, input_tensor)
        _pynative_executor.sync()


@log_function_entry_exit
def test_infer_hccl_all_gather_into_tensor_uneven():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    ## infer异常多维tensor
    # rank0: [0, 0], rank1: [[1, 1], [1, 1]], rank2: [[2, 2], [2, 2], [2, 2]], rank3: [[3, 3], [3, 3], [3, 3], [3, 3]]...
    input_tensor = ms.Tensor(np.ones([rank + 1, 2]).astype(np.float32) * rank)
    total_size = sum(r + 1 for r in range(size))
    output_tensor = ms.Tensor(np.zeros([total_size, 2]).astype(np.float32))
    output_split_sizes = [r + 1 for r in range(size)]
    with pytest.raises(ValueError):
        output_split_sizes1 = [r + 3 for r in range(size)]
        all_gather_into_tensor_uneven(
            output_tensor, input_tensor, output_split_sizes=output_split_sizes1
        )
        _pynative_executor.sync()
    with pytest.raises(ValueError):
        output_split_sizes1 = [r + 1 for r in range(size + 3)]
        all_gather_into_tensor_uneven(
            output_tensor, input_tensor, output_split_sizes=output_split_sizes1
        )
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([5 * size]).astype(np.float32))
    with pytest.raises(ValueError):
        all_gather_into_tensor_uneven(
            output_tensor, input_tensor, output_split_sizes=output_split_sizes
        )
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([total_size]).astype(np.int32))
    with pytest.raises(ValueError):
        all_gather_into_tensor_uneven(output_tensor, input_tensor)
        _pynative_executor.sync()


@log_function_entry_exit
def test_infer_hccl_reduce_scatter_tensor():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # infer异常
    input_tensor = ms.Tensor(np.ones([3 * size, 3]).astype(np.float32))
    output_tensor = ms.Tensor(np.zeros([1, 3]).astype(np.float32))
    with pytest.raises(ValueError):
        reduce_scatter_tensor(output_tensor, input_tensor)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 1]).astype(np.float32))
    with pytest.raises(ValueError):
        reduce_scatter_tensor(output_tensor, input_tensor)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.int32))
    with pytest.raises(ValueError):
        reduce_scatter_tensor(output_tensor, input_tensor)
        _pynative_executor.sync()


@log_function_entry_exit
def test_infer_hccl_reduce_scatter_tensor_uneven():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    ## infer异常多维tensor
    # input_tensor: [[0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2], ...]
    # rank0: [[0, 0]], rank1: [[1, 1], [1, 1]], rank2: [[2, 2], [2, 2], [2, 2]], rank3: [[3, 3], [3, 3], [3, 3
    # ], [3, 3]]...
    input_tensor = ms.Tensor(np.concatenate([np.ones([r + 1, 2]) * r for r in range(size)]).astype(np.float32))
    output_tensor = ms.Tensor(np.zeros([rank + 1, 2]).astype(np.float32))
    with pytest.raises(ValueError):
        input_split_sizes1 = [r + 3 for r in range(size)]
        reduce_scatter_tensor_uneven(
            output_tensor, input_tensor, input_split_sizes=input_split_sizes1
        )
        _pynative_executor.sync()
    with pytest.raises(ValueError):
        input_split_sizes1 = [r + 1 for r in range(size + 3)]
        reduce_scatter_tensor_uneven(
            output_tensor, input_tensor, input_split_sizes=input_split_sizes1
        )
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([rank + 1]).astype(np.int32))
    with pytest.raises(ValueError):
        reduce_scatter_tensor_uneven(output_tensor, input_tensor)
        _pynative_executor.sync()


@log_function_entry_exit
def test_infer_hccl_all_to_all():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # infer异常
    input_tensor = ms.Tensor(np.ones([1, 1]).astype(np.float32)) * rank
    input_tensors = []
    output_tensors = []
    for _ in range(size):
        input_tensors.append(input_tensor)
        output_tensors.append(ms.Tensor(np.ones([1, 1]).astype(np.int32)))
    with pytest.raises(ValueError):
        all_to_all(output_tensors, input_tensors)
        _pynative_executor.sync()


@log_function_entry_exit
def test_infer_hccl_all_to_all_single():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # infer异常
    with pytest.raises(ValueError):
        input_tensor = ms.Tensor(np.ones([size, 1]).astype(np.float32)) * rank
        output_tensor = ms.Tensor(np.zeros([size, 1]).astype(np.int32))
        all_to_all_single(output_tensor, input_tensor)
        _pynative_executor.sync()


@log_function_entry_exit
def test_infer_hccl_all_gather():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # infer异常
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    with pytest.raises(TypeError):
        output_tensor = []
        for _ in range(size):
            output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.int32)))
        all_gather(output_tensor, input_tensor)
        _pynative_executor.sync()


@log_function_entry_exit
def test_infer_hccl_reduce_scatter():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # infer异常
    input_tensor = []
    for _ in range(size):
        input_tensor.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))
    output_tensor = ms.Tensor(np.zeros([1, 3]).astype(np.float32))
    with pytest.raises(TypeError):
        reduce_scatter(output_tensor, input_tensor)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 1]).astype(np.float32))
    with pytest.raises(TypeError):
        reduce_scatter(output_tensor, input_tensor)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.int32))
    with pytest.raises(TypeError):
        reduce_scatter(output_tensor, input_tensor)
        _pynative_executor.sync()


@log_function_entry_exit
def test_infer_hccl_reduce_scatter_diff_shape():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # infer异常多维tensor
    # rank0: ([0, 0], [[0, 0], [0, 0]], ...)
    # rank1: ([1, 1], [[1, 1], [1, 1]], ...)
    # rank2: ([2, 2], [[2, 2], [2, 2]], ...)
    # output: ([sum, sum], [[sum, sum], [sum, sum]], ...)
    input_list = [ms.Tensor(np.ones([ii + 1, 2]) * rank, dtype=ms.int32) for ii in range(size)]
    # output tensor shape not match real op output.
    output_tensor = ms.Tensor(np.zeros([size+1]).astype(np.int32))
    with pytest.raises(TypeError):
        reduce_scatter(output_tensor, input_list)
        _pynative_executor.sync()


@log_function_entry_exit
def test_infer_hccl_gather():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # infer异常
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    with pytest.raises(TypeError):
        output_tensor = []
        for _ in range(size):
            output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.int32)))
        gather(input_tensor, output_tensor, dst=rank)
        _pynative_executor.sync()


@log_function_entry_exit
def test_infer_hccl_scatter():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # infer异常
    input_tensor = []
    for _ in range(size):
        input_tensor.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))
    if rank != 0:
        input_tensor = []
        for _ in range(size):
            input_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    with pytest.raises(TypeError):
        input_tensor1 = [
            ms.Tensor(np.zeros([3, 3]).astype(np.float32)),
            ms.Tensor(np.zeros([3, 3]).astype(np.int32)),
        ]
        scatter(output_tensor, input_tensor1)
        _pynative_executor.sync()
    with pytest.raises(TypeError):
        input_tensor1 = [
            ms.Tensor(np.zeros([3, 3]).astype(np.float32)),
            ms.Tensor(np.zeros([1, 3]).astype(np.float32)),
        ]
        scatter(output_tensor, input_tensor1)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([1, 3]).astype(np.float32))
    with pytest.raises(TypeError):
        scatter(output_tensor, input_tensor, src=rank)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 1]).astype(np.float32))
    with pytest.raises(TypeError):
        scatter(output_tensor, input_tensor, src=rank)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.int32))
    with pytest.raises(TypeError):
        scatter(output_tensor, input_tensor, src=rank)
        _pynative_executor.sync()
