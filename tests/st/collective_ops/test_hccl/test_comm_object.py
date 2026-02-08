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
import numpy as np
import pytest
import mindspore as ms
from mindspore import context
from mindspore.mint.distributed.distributed import (
    init_process_group,
    new_group,
    get_rank,
    get_world_size,
    all_gather_object,
    broadcast_object_list,
    gather_object,
    scatter_object_list,
)
#msrun --worker_num=8 --local_worker_num=8 --master_port=10923 --bind_core True --join True pytest -sv --disable-warnings  test_comm_object.py
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
def test_all_gather_object():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    #正常场景
    obj = str(rank)
    object_gather_list = []
    for _ in range(size):
        object_gather_list.append(None)
    all_gather_object(object_gather_list, obj)
    assert len(object_gather_list) == size
    for i in range(size):
        assert object_gather_list[i] == str(i)
    #异常用例
    with pytest.raises(TypeError):
        all_gather_object(object_gather_list, obj, group=1)
    with pytest.raises(TypeError):
        all_gather_object(None, obj)
    with pytest.raises(TypeError):
        all_gather_object({None}, obj)
    with pytest.raises(TypeError):
        all_gather_object({}, obj)
    with pytest.raises(TypeError):
        all_gather_object(1, obj)


@log_function_entry_exit
def test_broadcast_object_list():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    object_list = []
    for i in range(size):
        if rank != 0:
            object_list.append(None)
        else:
            object_list.append(str(i))
    broadcast_object_list(object_list)
    assert len(object_list) == size
    for i in range(size):
        assert object_list[i] == str(i)
    #异常用例
    with pytest.raises(TypeError):
        broadcast_object_list(object_list, group=1)
    with pytest.raises(TypeError):
        broadcast_object_list(object_list, src="1")
    with pytest.raises(TypeError):
        broadcast_object_list(None)
    with pytest.raises(TypeError):
        broadcast_object_list({})
    with pytest.raises(TypeError):
        broadcast_object_list(1)


@log_function_entry_exit
def test_gather_object():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    obj = str(rank)
    object_gather_list = []
    for _ in range(size):
        object_gather_list.append(None)
    gather_object(obj, object_gather_list)
    assert len(object_gather_list) == size
    if rank == 0:
        for i in range(size):
            assert object_gather_list[i] == str(i)
    #异常用例
    with pytest.raises(TypeError):
        gather_object(obj, object_gather_list, group=1)
    with pytest.raises(TypeError):
        gather_object(obj, object_gather_list, dst="1")
    if rank == 0:
        with pytest.raises(TypeError):
            gather_object(obj, None)
        with pytest.raises(TypeError):
            gather_object(obj, {})
        with pytest.raises(TypeError):
            gather_object(obj, 1)


@log_function_entry_exit
def test_hccl_scatter_object_list():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    scatter_object_input_list = []
    scatter_object_output_list = [None]
    for i in range(size):
        if rank != 0:
            scatter_object_input_list.append(None)
        else:
            scatter_object_input_list.append(str(i))
    scatter_object_list(scatter_object_output_list, scatter_object_input_list)
    assert len(scatter_object_input_list) == size
    assert scatter_object_output_list[0] == str(rank)
    if rank == 2 or rank == 3:
        group = new_group([2, 3])
        scatter_object_input_list = []
        scatter_object_output_list = [None]
        for i in range(2):
            if rank != 2:
                scatter_object_input_list.append(None)
            else:
                scatter_object_input_list.append(str(i))
        scatter_object_list(scatter_object_output_list, scatter_object_input_list, src=2, group=group)
        assert len(scatter_object_input_list) == 2
        assert scatter_object_output_list[0] == str(rank-2)
    #异常用例
    with pytest.raises(TypeError):
        scatter_object_list(scatter_object_output_list, scatter_object_input_list, group=1)
    with pytest.raises(TypeError):
        scatter_object_list(scatter_object_output_list, scatter_object_input_list, src="1")
    if rank == 0:
        with pytest.raises(TypeError):
            scatter_object_list(scatter_object_output_list, None)
        with pytest.raises(TypeError):
            scatter_object_list(scatter_object_output_list, {})
        with pytest.raises(TypeError):
            scatter_object_list(scatter_object_output_list, {None})
    with pytest.raises(TypeError):
        scatter_object_list(None, scatter_object_input_list)
    with pytest.raises(TypeError):
        scatter_object_list({}, scatter_object_input_list)
