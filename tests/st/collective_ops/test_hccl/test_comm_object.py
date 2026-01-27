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
    send_object_list,
    recv_object_list,
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

class Person:
    def __init__(self, name, person_id):
        self.name = name
        self.rank = person_id
    def greet(self):
        return f"Hello, I'm {self.name}, rank id is {self.rank}"

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




@log_function_entry_exit
def test_hccl_send_object_list():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    if rank % 2 == 0:
        objects = ["foo", 12, {1: 2}]  # any picklable object
        send_object_list(objects, dst=(rank + 1) % size)
    else:
        objects = [None, None, None]
        recv_object_list(objects, src=(rank - 1) % size)
        assert objects == ["foo", 12, {1: 2}]
    # group场景
    if rank == 2 or rank == 4:
        group = new_group([2,4])
        if rank == 4:
            objects = ["test", 42, {"key": "value"}]
            send_object_list(objects, dst=2, group=group)
        else:
            objects = [None, None, None]
            recv_object_list(objects, src=4, group=group)
            assert objects == ["test", 42, {"key": "value"}]
    # 异常场景
    with pytest.raises(TypeError):
        send_object_list(1, dst=1)
    with pytest.raises(TypeError):
        send_object_list(["test"], dst="test")
    with pytest.raises(TypeError):
        send_object_list(["test"], dst=1, group=1)
    with pytest.raises(ValueError):
        send_object_list(["test"], dst=rank)
    with pytest.raises(TypeError):
        recv_object_list(1, src=0)
    with pytest.raises(TypeError):
        recv_object_list(["test"], src="test")
    with pytest.raises(TypeError):
        recv_object_list(["test"], src=0, group=1)


@log_function_entry_exit
def test_hccl_recv_object_list():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    if rank % 2 == 0:
        objects = ["bar", 34, {"a": "b"}]  # any picklable object
        send_object_list(objects, dst=(rank + 1) % size)
    else:
        objects = [None, None, None]
        out = recv_object_list(objects, src=(rank - 1) % size)
        assert out == 0
        assert objects == ["bar", 34, {"a": "b"}]
    # group场景
    if rank == 3 or rank == 5:
        group = new_group([3,5])
        if rank == 5:
            objects = [1, 2, 3]
            send_object_list(objects, dst=3, group=group)
        else:
            objects = [None, None, None]
            out = recv_object_list(objects, src=5, group=group)
            assert out == 0
            assert objects == [1, 2, 3]
    # 异常场景
    with pytest.raises(TypeError):
        recv_object_list(1)
    with pytest.raises(TypeError):
        recv_object_list(["test"], src="test")
    with pytest.raises(TypeError):
        recv_object_list(["test"], group=1)


@log_function_entry_exit
def test_hccl_send_object_list_class():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    if rank % 2 == 0:
        # 创建对象
        person = Person("Alice", rank)
        objects = [person]  # any picklable object
        send_object_list(objects, dst=(rank + 1) % size)
    else:
        objects = [None]
        recv_object_list(objects, src=(rank - 1) % size)
        objects[0].greet()
        assert objects[0].name == "Alice"
        assert objects[0].rank == rank - 1
    # group场景
    if rank == 2 or rank == 4:
        group = new_group([2,4])
        if rank == 4:
            person = Person("Alice", rank)
            objects = [person, person]
            send_object_list(objects, dst=2, group=group)
        else:
            objects = [None, None]
            recv_object_list(objects, src=4, group=group)
            objects[1].greet()
            assert objects[1].name == "Alice"
            assert objects[1].rank == 4


@log_function_entry_exit
def test_hccl_all_gather_object_class():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    person = Person("Alice", rank)
    obj = person
    object_list = [None] * size
    all_gather_object(object_list, obj)
    # 验证每个rank都收到了所有对象
    for i in range(size):
        assert object_list[i].name == "Alice"
        assert object_list[i].rank == i
        assert object_list[i].greet() == f"Hello, I'm Alice, rank id is {i}"
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        person = Person("Bob", rank)
        obj = person
        object_list = [None, None]
        all_gather_object(object_list, obj, group=group)
        assert object_list[0].name == "Bob"
        assert object_list[0].rank == 0
        assert object_list[1].name == "Bob"
        assert object_list[1].rank == 1


@log_function_entry_exit
def test_hccl_broadcast_object_list_class():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    if rank == 0:
        person = Person("Alice", rank)
        objects = [person]
    else:
        objects = [None]
    broadcast_object_list(objects, src=0)
    assert objects[0].name == "Alice"
    assert objects[0].rank == 0
    assert objects[0].greet() == "Hello, I'm Alice, rank id is 0"
    # group场景
    if rank == 2 or rank == 4:
        group = new_group([2, 4])
        if rank == 2:
            person = Person("Charlie", rank)
            objects = [person]
        else:
            objects = [None]
        broadcast_object_list(objects, src=2, group=group)
        assert objects[0].name == "Charlie"
        assert objects[0].rank == 2


@log_function_entry_exit
def test_hccl_gather_object_class():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    person = Person("Alice", rank)
    obj = person
    if rank == 0:
        object_gather_list = [None] * size
        gather_object(obj, object_gather_list, dst=0)
        # 验证收集到的对象
        for i in range(size):
            assert object_gather_list[i].name == "Alice"
            assert object_gather_list[i].rank == i
    else:
        gather_object(obj, dst=0)
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        person = Person("Bob", rank)
        obj = person
        if rank == 1:
            object_gather_list = [None, None]
            gather_object(obj, object_gather_list, dst=1, group=group)
            assert object_gather_list[0].name == "Bob"
            assert object_gather_list[0].rank == 0
            assert object_gather_list[1].name == "Bob"
            assert object_gather_list[1].rank == 1
        else:
            gather_object(obj, dst=1, group=group)


@log_function_entry_exit
def test_hccl_scatter_object_list_class():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    if rank == 0:
        scatter_object_input_list = [Person("Alice", i) for i in range(size)]
        scatter_object_output_list = [None]
        scatter_object_list(scatter_object_output_list, scatter_object_input_list, src=0)
        assert scatter_object_output_list[0].name == "Alice"
        assert scatter_object_output_list[0].rank == 0
    else:
        scatter_object_output_list = [None]
        scatter_object_list(scatter_object_output_list, None, src=0)
        assert scatter_object_output_list[0].name == "Alice"
        assert scatter_object_output_list[0].rank == rank
    # group场景
    if rank == 2 or rank == 4:
        group = new_group([2, 4])
        if rank == 2:
            scatter_object_input_list = [Person("Bob", 2), Person("Bob", 4)]
            scatter_object_output_list = [None]
            scatter_object_list(scatter_object_output_list, scatter_object_input_list, src=2, group=group)
            assert scatter_object_output_list[0].name == "Bob"
            assert scatter_object_output_list[0].rank == 2
        else:
            scatter_object_output_list = [None]
            scatter_object_list(scatter_object_output_list, None, src=2, group=group)
            assert scatter_object_output_list[0].name == "Bob"
            assert scatter_object_output_list[0].rank == 4
