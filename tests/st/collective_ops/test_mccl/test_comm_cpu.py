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
The tests of mindspore, used to test communication for cpu.
"""
import os
import numpy as np
import hashlib
import time
import mindspore as ms
from mindspore import context
from mindspore.ops import ReduceOp
from mindspore.mint.distributed.distributed import (
    init_process_group,
    get_rank,
    get_world_size,
    get_backend,
    new_group,
    get_global_rank,
    get_process_group_ranks,
    broadcast,
    gather,
    scatter,
    all_gather,
    send,
    recv,
    barrier,
    all_reduce,
    TCPStore,
)
#msrun --worker_num=8 --local_worker_num=8 --master_port=10923 --bind_core True --join True pytest -sv --disable-warnings  test_comm_cpu.py
np.random.seed(1)

init_approach = os.environ.get('INIT_APPROACH')
node_id = get_rank()
worker_num = 8

if init_approach == "INIT_METHOD":
    print("Entry calling init_process_group with provided init_method!", flush=True)
    init_method = "tcp://127.0.0.1:10666"
    init_process_group(init_method=init_method, rank=node_id, world_size=worker_num)
elif init_approach == "TCPSTORE":
    print("Entry calling init_process_group with provided TcpStore!", flush=True)
    is_master = node_id == 0
    store = TCPStore("127.0.0.1", 10666, worker_num, is_master)
    init_process_group(rank=node_id, world_size=worker_num, store=store)
else:
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
def test_cpu_new_group():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    group = new_group(list(range(size)))
    name = "hccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    assert group == name
    group = new_group(list(range(size)), backend="hccl")
    assert group == name
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name
    group = new_group(list(range(size)), backend="mccl")
    assert group == name

    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name


@log_function_entry_exit
def test_cpu_get_rank():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    local_rank = get_rank()
    assert local_rank == rank
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        local_rank = get_rank(group)
        global_rank = get_global_rank(group, local_rank)
        assert global_rank == rank
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        local_rank = get_rank(group)
        global_rank = get_global_rank(group, local_rank)
        assert global_rank == rank


@log_function_entry_exit
def test_cpu_get_backend():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    backend = get_backend()
    assert backend == "hccl"
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        backend = get_backend(group)
        assert backend == "mccl"
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        backend = get_backend(group)
        assert backend == "mccl"


@log_function_entry_exit
def test_cpu_get_process_group_ranks():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    ranks = get_process_group_ranks()
    print("ranks is:", ranks)
    assert ranks == list(range(size))
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        ranks = get_process_group_ranks(group)
        assert ranks == list(range(2))
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        ranks = get_process_group_ranks(group)
        assert ranks == [2, 3]


@log_function_entry_exit
def test_cpu_broadcast():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name
    # 同步场景
    tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    if rank != 0:
        tensor = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    output_handle = broadcast(tensor, src=0, group=group)
    assert output_handle is None
    except_output_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    assert np.allclose(tensor.asnumpy(), except_output_tensor.asnumpy())

    # 异步场景
    tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    if rank != 1:
        tensor = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    output_handle = broadcast(tensor, src=1, group=group, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    except_output_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    assert np.allclose(tensor.asnumpy(), except_output_tensor.asnumpy())

    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
        if rank != 0:
            tensor = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
        output_handle = broadcast(tensor, src=0, group=name)
        assert output_handle is None
        except_output_tensor = ms.Tensor(
            np.arange(8).reshape([2, 4]).astype(np.float32)
        )
        assert np.allclose(tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, [2, 3])), "utf-8")).hexdigest()
        assert group == name
        tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.int32))
        if rank != 2:
            tensor = ms.Tensor(np.zeros([2, 4]).astype(np.int32))
        output_handle = broadcast(tensor, src=2, group=name)
        assert output_handle is None
        except_output_tensor = ms.Tensor(
            np.arange(8).reshape([2, 4]).astype(np.int32)
        )
        assert np.allclose(tensor.asnumpy(), except_output_tensor.asnumpy())


@log_function_entry_exit
def test_cpu_broadcast_dtype():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    def test_cpu_broadcast_dtype_inner(dtype=np.float32):
        """
        Feature: test distributed op
        Description: test broadcast dtype
        Expectation: success
        """
        name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
        group = new_group(list(range(size)), backend="mccl")
        assert group == name
        # 同步场景
        tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(dtype))
        if rank != 0:
            tensor = ms.Tensor(np.zeros([2, 4]).astype(dtype))
        output_handle = broadcast(tensor, src=0, group=group)
        assert output_handle is None
        except_output_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(dtype))
        assert np.allclose(tensor.asnumpy(), except_output_tensor.asnumpy())


    types = [np.int8, np.uint8, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float16, np.float32, np.float64]
    for type_i in types:
        test_cpu_broadcast_dtype_inner(type_i)


@log_function_entry_exit
def test_cpu_all_gather():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name
    # 同步场景
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_tensor = []
    except_output_tensor = []
    for _ in range(size):
        output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
        except_output_tensor.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))

    output_handle = all_gather(output_tensor, input_tensor, group=group)
    assert output_handle is None
    assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
    assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())
    # 异步场景
    output_tensor = []
    for _ in range(size):
        output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
    output_handle = all_gather(output_tensor, input_tensor, group=group, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
    assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
        output_tensor1 = []
        except_output_tensor = []
        for _ in range(2):
            output_tensor1.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
            except_output_tensor.append(input_tensor1)
        output_handle = all_gather(output_tensor1, input_tensor1, group=name)
        assert output_handle is None
        assert np.allclose(
            output_tensor1[0].asnumpy(), except_output_tensor[0].asnumpy()
        )
        assert np.allclose(
            output_tensor1[1].asnumpy(), except_output_tensor[1].asnumpy()
        )
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, [2, 3])), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.int64))
        output_tensor1 = []
        except_output_tensor = []
        for _ in range(2):
            output_tensor1.append(ms.Tensor(np.zeros([3, 3]).astype(np.int64)))
            except_output_tensor.append(input_tensor1)
        output_handle = all_gather(output_tensor1, input_tensor1, group=name)
        assert output_handle is None
        assert np.allclose(
            output_tensor1[0].asnumpy(), except_output_tensor[0].asnumpy()
        )
        assert np.allclose(
            output_tensor1[1].asnumpy(), except_output_tensor[1].asnumpy()
        )


@log_function_entry_exit
def test_cpu_all_gather_dtype():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    def test_cpu_all_gather_dtype_inner(dtype=np.float32):
        """
        Feature: test distributed op
        Description: test allgather dtype
        Expectation: success
        """
        name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
        group = new_group(list(range(size)), backend="mccl")
        assert group == name
        # 同步场景
        input_tensor = ms.Tensor(np.ones([3, 3]).astype(dtype))
        output_tensor = []
        except_output_tensor = []
        for _ in range(size):
            output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(dtype)))
            except_output_tensor.append(ms.Tensor(np.ones([3, 3]).astype(dtype)))

        output_handle = all_gather(output_tensor, input_tensor, group=group)
        assert output_handle is None
        assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
        assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())

    types = [np.int8, np.uint8, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float16, np.float32, np.float64]
    for type_i in types:
        test_cpu_all_gather_dtype_inner(type_i)


@log_function_entry_exit
def test_cpu_gather():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name
    # 同步场景
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_tensor = []
    except_output_tensor = []
    for _ in range(size):
        output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
        if rank == 0:
            except_output_tensor.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))
        else:
            except_output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
    output_handle = gather(input_tensor, output_tensor, group=group)
    assert output_handle is None
    assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
    assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())
    # 异步场景
    output_tensor = []
    for _ in range(size):
        output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
    output_handle = gather(input_tensor, output_tensor, group=group, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
    assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
        output_tensor1 = []
        except_output_tensor = []
        for _ in range(2):
            output_tensor1.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
            if rank == 1:
                except_output_tensor.append(input_tensor1)
            else:
                except_output_tensor.append(
                    ms.Tensor(np.zeros([3, 3]).astype(np.float32))
                )
        output_handle = gather(input_tensor1, output_tensor1, dst=1, group=name)
        assert output_handle is None
        assert np.allclose(
            output_tensor1[0].asnumpy(), except_output_tensor[0].asnumpy()
        )
        assert np.allclose(
            output_tensor1[1].asnumpy(), except_output_tensor[1].asnumpy()
        )
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, [2, 3])), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float16))
        output_tensor1 = []
        except_output_tensor = []
        for _ in range(2):
            output_tensor1.append(ms.Tensor(np.zeros([3, 3]).astype(np.float16)))
            if rank == 3:
                except_output_tensor.append(input_tensor1)
            else:
                except_output_tensor.append(
                    ms.Tensor(np.zeros([3, 3]).astype(np.float16))
                )
        output_handle = gather(input_tensor1, output_tensor1, dst=3, group=name)
        assert output_handle is None
        assert np.allclose(
            output_tensor1[0].asnumpy(), except_output_tensor[0].asnumpy()
        )
        assert np.allclose(
            output_tensor1[1].asnumpy(), except_output_tensor[1].asnumpy()
        )


@log_function_entry_exit
def test_cpu_gather_dtype():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    def test_cpu_gather_dtype_inner(dtype=np.float32):
        """
        Feature: test distributed op
        Description: test gather dtype
        Expectation: success
        """
        name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
        group = new_group(list(range(size)), backend="mccl")
        assert group == name
        # 同步场景
        input_tensor = ms.Tensor(np.ones([3, 3]).astype(dtype))
        output_tensor = []
        except_output_tensor = []
        for _ in range(size):
            output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(dtype)))
            if rank == 0:
                except_output_tensor.append(ms.Tensor(np.ones([3, 3]).astype(dtype)))
            else:
                except_output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(dtype)))
        output_handle = gather(input_tensor, output_tensor, group=group)
        assert output_handle is None
        assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
        assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())

    types = [np.int8, np.uint8, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float16, np.float32, np.float64]
    for type_i in types:
        test_cpu_gather_dtype_inner(type_i)


@log_function_entry_exit
def test_cpu_scatter():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name
    # 同步场景
    input_tensor = []
    for _ in range(size):
        input_tensor.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))
    if rank != 0:
        input_tensor = []
        for _ in range(size):
            input_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_handle = scatter(output_tensor, input_tensor, src=0, group=group)
    assert output_handle is None
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # 异步场景
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    output_handle = scatter(output_tensor, input_tensor, src=0, group=group, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = []
        for _ in range(2):
            input_tensor1.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))
        output_tensor1 = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
        output_handle = scatter(output_tensor1, input_tensor1, src=0, group=name)
        except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, [2, 3])), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = []
        for _ in range(2):
            input_tensor1.append(ms.Tensor(np.ones([3, 3]).astype(np.int8)))
        output_tensor1 = ms.Tensor(np.zeros([3, 3]).astype(np.int8))
        output_handle = scatter(output_tensor1, input_tensor1, src=2, group=name)
        except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.int8))
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())


@log_function_entry_exit
def test_cpu_scatter_dtype():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    def test_cpu_scatter_dtype_inner(dtype=np.float32):
        """
        Feature: test distributed op
        Description: test scatter dtype
        Expectation: success
        """
        name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
        group = new_group(list(range(size)), backend="mccl")
        assert group == name
        # 同步场景
        input_tensor = []
        for _ in range(size):
            input_tensor.append(ms.Tensor(np.ones([3, 3]).astype(dtype)))
        if rank != 0:
            input_tensor = []
            for _ in range(size):
                input_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(dtype)))
        output_tensor = ms.Tensor(np.zeros([3, 3]).astype(dtype))
        except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(dtype))
        output_handle = scatter(output_tensor, input_tensor, src=0, group=group)
        assert output_handle is None
        assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())

    types = [np.int8, np.uint8, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float16, np.float32, np.float64]
    for type_i in types:
        test_cpu_scatter_dtype_inner(type_i)



@log_function_entry_exit
def test_cpu_barrier():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name

    output_handle = barrier(group=group)
    assert output_handle is None
    # 异步场景
    output_handle = barrier(group=group, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    # group场景
    if rank == 2 or rank == 3:
        group = new_group([2, 3], 1, backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, [2, 3])), "utf-8")).hexdigest()
        assert group == name
        output_handle = barrier(group=group)
        assert output_handle is None


@log_function_entry_exit
def test_cpu_barrier1():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    a = ms.Tensor(np.random.randn(2, 3).astype(np.float32))
    b = ms.Tensor(np.random.randn(2, 3).astype(np.float32))
    ranks_list1 = [0, 1, 2, 3]
    ranks_list2 = [4, 5, 6, 7]
    if rank in ranks_list1:
        mccl_group = new_group(ranks_list1, backend="mccl")
        res = barrier(group=mccl_group, async_op=False)
        assert res is None
        start = time.time()
        if rank in (0, 1):
            time.sleep(2)
        c = a + b
        print(c)
        res = barrier(group=mccl_group, async_op=False)
        assert res is None
        end = time.time()
        t = end - start
        assert 2 < t < 3
    else:
        mccl_group = new_group(ranks_list2, backend="mccl")
        res = barrier(group=mccl_group, async_op=False)
        assert res is None
        start = time.time()
        if rank in (5, 6):
            time.sleep(4)
        c = a + b
        print(c)
        res = barrier(group=mccl_group, async_op=False)
        assert res is None
        end = time.time()
        t = end - start
        assert 4 < t < 5


@log_function_entry_exit
def test_cpu_send():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name

    input_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    output = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    if rank % 2 == 0:
        send(input_tensor, rank + 1 % size, group=group)
    else:
        out = recv(output, src=rank - 1, group=group)
        assert out == 0
        assert np.allclose(output.asnumpy(), input_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1, backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        if rank == 1:
            send(input_tensor, dst=0, group=group)
        else:
            out = recv(output, src=1, group=group)
            assert out == 0
            assert np.allclose(output.asnumpy(), input_tensor.asnumpy())


@log_function_entry_exit
def test_cpu_send_dtype():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    def test_cpu_send_dtype_inner(dtype=np.float32):
        """
        Feature: test distributed op
        Description: test send dtype
        Expectation: success
        """
        # 同步场景
        name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
        group = new_group(list(range(size)), backend="mccl")
        assert group == name

        input_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(dtype))
        output = ms.Tensor(np.zeros([2, 4]).astype(dtype))
        if rank % 2 == 0:
            send(input_tensor, rank + 1 % size, group=group)
        else:
            out = recv(output, src=rank - 1, group=group)
            assert out == 0
            assert np.allclose(output.asnumpy(), input_tensor.asnumpy())

    types = [np.int8, np.uint8, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float16, np.float32, np.float64]
    for type_i in types:
        test_cpu_send_dtype_inner(type_i)


@log_function_entry_exit
def test_cpu_recv():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name

    input_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    output = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    if rank % 2 == 0:
        send(input_tensor, rank + 1 % size, group=group)
    else:
        out = recv(output, src=rank - 1, group=group)
        assert out == 0
        assert np.allclose(output.asnumpy(), input_tensor.asnumpy())


@log_function_entry_exit
def test_cpu_all_reduce_type():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name

    input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))

    sum_input_tensor = input_tensor * (rank + 1)
    sum_output_handle = all_reduce(sum_input_tensor, op=ReduceOp.SUM, group=group)
    assert sum_output_handle is None
    except_sum_output = input_tensor * (sum(list(range(1, size + 1))))

    max_input_tensor = input_tensor * (rank + 1)
    max_output_handle = all_reduce(max_input_tensor, op=ReduceOp.MAX, group=group)
    assert max_output_handle is None
    except_max_output = input_tensor * size

    min_input_tensor = input_tensor * (rank + 1)
    min_output_handle = all_reduce(min_input_tensor, op=ReduceOp.MIN, group=group)
    assert min_output_handle is None
    except_min_output = input_tensor

    prod_input_tensor = input_tensor * 1
    prod_output_handle = all_reduce(prod_input_tensor, op=ReduceOp.PROD, group=group)
    assert prod_output_handle is None
    except_prod_output = input_tensor ** size

    assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
    assert np.allclose(max_input_tensor.asnumpy(), except_max_output.asnumpy())
    assert np.allclose(min_input_tensor.asnumpy(), except_min_output.asnumpy())
    assert np.allclose(prod_input_tensor.asnumpy(), except_prod_output.asnumpy())


@log_function_entry_exit
def test_cpu_all_reduce():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name
    input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
    # 同步场景
    sum_input_tensor = input_tensor * (rank + 1)
    sum_output_handle = all_reduce(sum_input_tensor, group=group)
    except_sum_output = input_tensor * (sum(list(range(1, size + 1))))
    assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
    assert sum_output_handle is None
    # 异步场景
    sum_input_tensor = input_tensor * (rank + 1)
    sum_output_handle = all_reduce(sum_input_tensor, group=group, async_op=True)
    assert sum_output_handle is not None
    sum_output_handle.wait()
    assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1, backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.int32))
        # 同步场景
        sum_input_tensor = input_tensor * (rank + 1)
        sum_output_handle = all_reduce(sum_input_tensor, group=group)
        except_sum_output = input_tensor * (sum(list(range(1, 3))))
        assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
        assert sum_output_handle is None
    if rank == 2 or rank == 3:
        group = new_group([2, 3], 1, backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, [2, 3])), "utf-8")).hexdigest()
        assert group == name
        input_tensor = ms.Tensor(np.arange(1024).reshape(32, 32).astype(np.int32))
        # 同步场景
        sum_input_tensor = input_tensor * (rank + 1)
        sum_output_handle = all_reduce(sum_input_tensor, group=group)
        except_sum_output = input_tensor * (sum(list(range(3, 5))))
        assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
        assert sum_output_handle is None


@log_function_entry_exit
def test_cpu_all_reduce_dtype():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    def test_cpu_all_reduce_dtype_inner(dtype=np.float32):
        """
        Feature: test distributed op
        Description: test allreduce dtype
        Expectation: success
        """
        name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
        group = new_group(list(range(size)), backend="mccl")
        assert group == name
        input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(dtype))
        # 同步场景
        sum_input_tensor = input_tensor * (rank + 1)
        sum_output_handle = all_reduce(sum_input_tensor, group=group)
        except_sum_output = input_tensor * (sum(list(range(1, size + 1))))
        assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
        assert sum_output_handle is None

    types = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32]
    for type_i in types:
        test_cpu_all_reduce_dtype_inner(type_i)
