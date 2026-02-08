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
import hashlib
import mindspore as ms
from mindspore import context
from mindspore.ops import ReduceOp, cat
from mindspore.ops.communication import (
    init_process_group,
    get_rank,
    get_world_size,
    new_group,
    all_reduce,
    all_gather_into_tensor,
    all_to_all,
    all_to_all_single,
    reduce_scatter_tensor,
    isend,
    irecv,
    send,
    recv,
    barrier,
    set_comm_ops_inplace,
)

#msrun --worker_num=8 --local_worker_num=8 --master_port=10923 --bind_core True --join True --cluster_time_out=800  pytest -sv --disable-warnings test_comm_func_noninplace.py
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
set_comm_ops_inplace(False)

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
def test_hccl_all_reduce_type():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))

    sum_input_tensor = input_tensor * (rank + 1)
    sum_output_tensor, sum_output_handle = all_reduce(sum_input_tensor, op=ReduceOp.SUM)
    assert sum_output_handle is None
    except_sum_output = input_tensor * (sum(list(range(1, size + 1))))

    max_input_tensor = input_tensor * (rank + 1)
    max_output_tensor, max_output_handle = all_reduce(max_input_tensor, op=ReduceOp.MAX)
    assert max_output_handle is None
    except_max_output = input_tensor * size

    min_input_tensor = input_tensor * (rank + 1)
    min_output_tensor, min_output_handle = all_reduce(min_input_tensor, op=ReduceOp.MIN)
    assert min_output_handle is None
    except_min_output = input_tensor

    prod_input_tensor = input_tensor * 1
    prod_output_tensor, prod_output_handle = all_reduce(prod_input_tensor, op=ReduceOp.PROD)
    assert prod_output_handle is None
    except_prod_output = input_tensor ** size

    assert np.allclose(sum_output_tensor.asnumpy(), except_sum_output.asnumpy())
    assert np.allclose(max_output_tensor.asnumpy(), except_max_output.asnumpy())
    assert np.allclose(min_output_tensor.asnumpy(), except_min_output.asnumpy())
    assert np.allclose(prod_output_tensor.asnumpy(), except_prod_output.asnumpy())


@log_function_entry_exit
def test_hccl_all_reduce():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
    # 同步场景
    sum_input_tensor = input_tensor * (rank + 1)
    sum_output_tensor, sum_output_handle = all_reduce(sum_input_tensor)
    except_sum_output = input_tensor * (sum(list(range(1, size + 1))))
    assert np.allclose(sum_output_tensor.asnumpy(), except_sum_output.asnumpy())
    assert sum_output_handle is None
    # 异步场景
    sum_input_tensor = input_tensor * (rank + 1)
    sum_output_tensor, sum_output_handle = all_reduce(sum_input_tensor, async_op=True)
    assert sum_output_handle is not None
    sum_output_handle.wait()
    assert np.allclose(sum_output_tensor.asnumpy(), except_sum_output.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        sum_input_tensor1 = input_tensor * (rank + 1)
        sum_output_tensor1, sum_output_handle = all_reduce(sum_input_tensor1, group=name)
        except_sum_output = input_tensor * (sum(list(range(1, 3))))
        assert np.allclose(sum_output_tensor1.asnumpy(), except_sum_output.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        all_reduce(1)
    with pytest.raises(TypeError):
        all_reduce(sum_input_tensor, op=1)
    with pytest.raises(TypeError):
        all_reduce(sum_input_tensor, op="test")
    with pytest.raises(TypeError):
        all_reduce(sum_input_tensor, group=1)
    with pytest.raises(TypeError):
        all_reduce(sum_input_tensor, async_op="test")


@log_function_entry_exit
def test_hccl_all_gather_into_tensor():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    except_output_tensor = ms.Tensor(np.ones([3 * size, 3]).astype(np.float32))
    output_tensor, output_handle = all_gather_into_tensor(None, input_tensor)
    assert output_handle is None
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # 异步场景
    output_tensor, output_handle = all_gather_into_tensor(None, input_tensor, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
        output_tensor1, output_handle = all_gather_into_tensor(
            None, input_tensor1, group=name
        )
        except_output_tensor = cat([input_tensor1, input_tensor1])
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        all_gather_into_tensor(1)
    with pytest.raises(TypeError):
        all_gather_into_tensor(None, input_tensor, group=1)
    with pytest.raises(TypeError):
        all_gather_into_tensor(None, input_tensor, async_op="test")
    with pytest.raises(TypeError):
        all_gather_into_tensor(None, [1])


@log_function_entry_exit
def test_hccl_reduce_scatter_tensor_type():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    input_tensor = ms.Tensor(np.ones([3 * size, 3]).astype(np.float32))
    except_sum_output = ms.Tensor(np.ones([3, 3]).astype(np.float32)) * size
    sum_output_tensor, sum_output_handle = reduce_scatter_tensor(
        None, input_tensor, op=ReduceOp.SUM
    )
    assert sum_output_handle is None
    input_tensor = ms.Tensor(np.ones([3 * size, 3]).astype(np.float32)) * (rank + 1)
    except_max_output = ms.Tensor(np.ones([3, 3]).astype(np.float32)) * size
    max_output_tensor, sum_output_handle = reduce_scatter_tensor(
        None, input_tensor, op=ReduceOp.MAX
    )
    assert sum_output_handle is None
    input_tensor = ms.Tensor(np.ones([3 * size, 3]).astype(np.float32)) * (rank + 1)
    except_min_output = ms.Tensor(np.ones([3, 3]).astype(np.float32)) * 1
    min_output_tensor, sum_output_handle = reduce_scatter_tensor(
        None, input_tensor, op=ReduceOp.MIN
    )
    assert sum_output_handle is None
    assert np.allclose(sum_output_tensor.asnumpy(), except_sum_output.asnumpy())
    assert np.allclose(max_output_tensor.asnumpy(), except_max_output.asnumpy())
    assert np.allclose(min_output_tensor.asnumpy(), except_min_output.asnumpy())


@log_function_entry_exit
def test_hccl_reduce_scatter_tensor():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([3 * size, 3]).astype(np.float32))
    except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32)) * size
    output_tensor, output_handle = reduce_scatter_tensor(None, input_tensor)
    assert output_handle is None
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # 异步场景
    output_tensor, output_handle = reduce_scatter_tensor(None, input_tensor, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.ones([3 * 2, 3]).astype(np.float32))
        output_tensor1, output_handle = reduce_scatter_tensor(None, input_tensor1, group=name)
        except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32)) * 2
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        reduce_scatter_tensor(1)
    with pytest.raises(TypeError):
        reduce_scatter_tensor(None, input_tensor, op=1)
    with pytest.raises(TypeError):
        reduce_scatter_tensor(None, input_tensor, op="test")
    with pytest.raises(TypeError):
        reduce_scatter_tensor(None, input_tensor, group=1)
    with pytest.raises(TypeError):
        reduce_scatter_tensor(None, input_tensor, async_op="test")
    with pytest.raises(TypeError):
        reduce_scatter_tensor(None, [1])


@log_function_entry_exit
def test_hccl_barrier():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    output_handle = barrier()
    assert output_handle is None
    # 异步场景
    output_handle = barrier(async_op=True)
    assert output_handle is not None
    output_handle.wait()
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        output_handle = barrier(group=name)
        assert output_handle is None
    # 异常场景
    with pytest.raises(TypeError):
        barrier(group=1)
    with pytest.raises(TypeError):
        barrier(async_op="test")


@log_function_entry_exit
def test_hccl_send():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    output = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    if rank % 2 == 0:
        send(input_tensor, rank + 1 % size)
    else:
        out = recv(output, src=rank - 1)
        assert np.allclose(out.asnumpy(), input_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        if rank == 1:
            send(input_tensor, dst=0, group=group)
        else:
            out = recv(output, src=1, group=group)
            assert np.allclose(out.asnumpy(), input_tensor.asnumpy())

    # 异常场景
    with pytest.raises(TypeError):
        send(1)
    with pytest.raises(TypeError):
        send(input_tensor, dst="test")
    with pytest.raises(TypeError):
        send(input_tensor, group=1)
    with pytest.raises(ValueError):
        send(input_tensor, dst=rank)


@log_function_entry_exit
def test_hccl_recv():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    output = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    if rank % 2 == 0:
        send(input_tensor, rank + 1 % size)
    else:
        out = recv(output, src=rank - 1)
        assert np.allclose(out.asnumpy(), input_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        if rank == 1:
            send(input_tensor, dst=0, group=group)
        else:
            out = recv(output, src=1, group=group)
            assert np.allclose(out.asnumpy(), input_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        recv(1)
    with pytest.raises(TypeError):
        recv(output, src="test")
    with pytest.raises(TypeError):
        recv(output, group=1)


@log_function_entry_exit
def test_hccl_isend():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 异步场景
    input_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    output = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    if rank % 2 == 0:
        handle = isend(input_tensor, rank + 1 % size)
        assert handle is not None
        handle.wait()
    else:
        out = recv(output, src=rank - 1)
        assert np.allclose(out.asnumpy(), input_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        if rank == 1:
            handle = isend(input_tensor, dst=0, group=group)
            assert handle is not None
            handle.wait()
        else:
            out = recv(output, src=1, group=group)
            assert np.allclose(out.asnumpy(), input_tensor.asnumpy())

    # 异常场景
    with pytest.raises(TypeError):
        isend(1)
    with pytest.raises(TypeError):
        isend(input_tensor, dst="test")
    with pytest.raises(TypeError):
        isend(input_tensor, group=1)
    with pytest.raises(ValueError):
        isend(input_tensor, dst=rank)


@log_function_entry_exit
def test_hccl_irecv():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 异步场景
    input_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    output = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    if rank % 2 == 0:
        send(input_tensor, rank + 1 % size)
    else:
        out, handle = irecv(output, src=rank - 1)
        assert handle is not None
        handle.wait()
        assert np.allclose(out.asnumpy(), input_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        if rank == 1:
            send(input_tensor, dst=0, group=group)
        else:
            out, handle = irecv(output, src=1, group=group)
            assert handle is not None
            handle.wait()
            assert np.allclose(out.asnumpy(), input_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        irecv(1)
    with pytest.raises(TypeError):
        irecv(output, src="test")
    with pytest.raises(TypeError):
        irecv(output, group=1)


@log_function_entry_exit
def test_hccl_all_to_all():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([1, 1]).astype(np.float32)) * rank
    input_tensors = []
    output_tensors = []
    except_output_tensors = []
    for i in range(size):
        input_tensors.append(input_tensor)
        output_tensors.append(ms.Tensor(np.zeros([1, 1]).astype(np.float32)))
        except_output_tensors.append(ms.Tensor(np.ones([1, 1]).astype(np.float32)) * i)

    output, handle = all_to_all(output_tensors, input_tensors)
    assert handle is None
    assert np.allclose(output[0].asnumpy(), except_output_tensors[0].asnumpy())
    assert np.allclose(output[1].asnumpy(), except_output_tensors[1].asnumpy())
    # 异步场景

    except_output_tensors = []
    for i in range(size):
        output_tensors.append(ms.Tensor(np.zeros([1, 1]).astype(np.float32)))
        except_output_tensors.append(ms.Tensor(np.ones([1, 1]).astype(np.float32)) * i)

    output, handle = all_to_all(output_tensors, input_tensors, async_op=True)
    assert handle is not None
    handle.wait()
    assert np.allclose(output[0].asnumpy(), except_output_tensors[0].asnumpy())
    assert np.allclose(output[1].asnumpy(), except_output_tensors[1].asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        if rank == 0:
            send_tensor_list = [ms.Tensor(1.0), ms.Tensor([[2, 3], [4, 5.0]])]
            recv_tensor_list = [ms.Tensor((0), dtype=ms.float32), ms.Tensor([0, 0.0])]
            output, handle = all_to_all(recv_tensor_list, send_tensor_list, group=group)
            assert handle is None
            except_output_tensor = [
                ms.Tensor((1), dtype=ms.float32),
                ms.Tensor([2, 2.0]),
            ]
            assert np.allclose(
                output[0].asnumpy(), except_output_tensor[0].asnumpy()
            )
            assert np.allclose(
                output[1].asnumpy(), except_output_tensor[1].asnumpy()
            )
        if rank == 1:
            send_tensor_list = [ms.Tensor([2, 2.0]), ms.Tensor([4, 5, 6, 7.0])]
            recv_tensor_list = [
                ms.Tensor([[0, 0.0], [0, 0]]),
                ms.Tensor([0, 0, 0, 0.0]),
            ]
            output, handle = all_to_all(recv_tensor_list, send_tensor_list, group=group)
            assert handle is None
            except_output_tensor = [
                ms.Tensor([[2, 3.0], [4, 5]]),
                ms.Tensor([4, 5, 6, 7.0]),
            ]
            assert np.allclose(
                output[0].asnumpy(), except_output_tensor[0].asnumpy()
            )
            assert np.allclose(
                output[1].asnumpy(), except_output_tensor[1].asnumpy()
            )
    # 异常场景
    with pytest.raises(TypeError):
        all_to_all(1)
    with pytest.raises(TypeError):
        all_to_all(output_tensors, 1)
    with pytest.raises(TypeError):
        all_to_all(output_tensors, input_tensors, group=1)
    with pytest.raises(TypeError):
        all_to_all(output_tensors, input_tensors, async_op="1")


@log_function_entry_exit
def test_hccl_all_to_all_single():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([size, 1]).astype(np.float32)) * rank
    output_tensor = ms.Tensor(np.zeros([size, 1]).astype(np.float32))
    output, handle = all_to_all_single(output_tensor, input_tensor)
    assert handle is None
    except_output_tensor = ms.Tensor(
        np.arange(size).reshape([size, 1]).astype(np.float32)
    )
    assert np.allclose(output.asnumpy(), except_output_tensor.asnumpy())
    # 异步场景
    output_tensor = ms.Tensor(np.zeros([size, 1]).astype(np.float32))
    output, handle = all_to_all_single(output_tensor, input_tensor, async_op=True)
    assert handle is not None
    handle.wait()
    assert np.allclose(output.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        if rank == 0:
            tensor = ms.Tensor([[0, 1.0, 2.0], [3, 4, 5], [6, 7, 8], [0, 0, 0]])
            output = ms.Tensor(np.zeros([4, 3]).astype(np.float32))
            output, handle = all_to_all_single(output, tensor, [3, 1], [3, 1], group=group)
            assert handle is None
            except_output_tensor = ms.Tensor(
                [[0, 1.0, 2.0], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
            )
            assert np.allclose(output.asnumpy(), except_output_tensor.asnumpy())
        if rank == 1:
            tensor = ms.Tensor([[9, 10.0, 11], [12.0, 13, 14], [1, 1, 1]])
            output = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
            output, handle = all_to_all_single(output, tensor, [1, 2], [1, 2], group=group)
            assert handle is None
            except_output_tensor = ms.Tensor([[0, 0, 0.0], [12, 13, 14], [1, 1, 1]])
            assert np.allclose(output.asnumpy(), except_output_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        all_to_all_single(output_tensor, 1)
    with pytest.raises(TypeError):
        all_to_all_single(output_tensor, input_tensor, group=1)
    with pytest.raises(TypeError):
        all_to_all_single(output_tensor, input_tensor, async_op="1")
