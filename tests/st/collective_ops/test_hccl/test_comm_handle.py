# Copyright 2026 Huawei Technologies Co., Ltd
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
import time
import pytest
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore.runtime import Stream
from mindspore.mint.distributed.distributed import (
    init_process_group,
    get_rank,
    get_world_size,
    all_gather_into_tensor,
)

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
def test_comm_handle_destroy_after_wait1():
    """
    Feature: test comm handle destroy after wait
    Description: test comm handle destroy after wait
    Expectation: success
    """
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_tensor = ms.Tensor(np.zeros([24, 3]).astype(np.float32))
    for _ in range(10):
        comm_handle = all_gather_into_tensor(output_tensor, input_tensor, async_op=True)
        comm_handle.wait()
        comm_handle = None
    time.sleep(1)
    ms.runtime.synchronize()
    assert np.allclose(output_tensor.asnumpy(), np.ones([24, 3]).astype(np.float32))


@log_function_entry_exit
def test_comm_handle_destroy_after_wait2():
    """
    Feature: test comm handle destroy after wait
    Description: test comm handle destroy after wait
    Expectation: success
    """
    tensor0 = ms.Tensor(np.zeros([24, 3]).astype(np.float32))
    tensor1 = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    input_tensor = None
    output_tensor = None
    s1 = Stream()
    s2 = Stream()
    with ms.runtime.StreamCtx(s1):
        input_tensor = tensor1 * 1.0
        output_tensor = tensor0 * 1.0
    s2.wait_stream(s1)
    with ms.runtime.StreamCtx(s2):
        for _ in range(10):
            comm_handle = all_gather_into_tensor(output_tensor, input_tensor, async_op=True)
            comm_handle.wait()
            comm_handle = None
    time.sleep(1)
    ms.runtime.synchronize()
    assert np.allclose(output_tensor.asnumpy(), np.ones([24, 3]).astype(np.float32))



@log_function_entry_exit
def test_comm_handle_destroy_before_wait1():
    """
    Feature: test comm handle destroy before wait
    Description: test comm handle destroy before wait
    Expectation: success
    """
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_tensor = ms.Tensor(np.zeros([24, 3]).astype(np.float32))
    with pytest.raises(RuntimeError):
        for _ in range(10):
            comm_handle = all_gather_into_tensor(output_tensor, input_tensor, async_op=True)
            del comm_handle
        time.sleep(1)
        ms.runtime.synchronize()
