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

import numpy as np
import pytest
import hashlib
import mindspore as ms
from mindspore.communication import init, create_group
from mindspore.communication.comm_func import all_to_all_v_c
from mindspore.communication.management import get_rank, get_group_size
from mindspore.graph.api import _pynative_executor

# 'all_to_all_single_with_output_shape' function only supports KernelByKernel mode by now.
np.random.seed(1)
ms.set_context(jit_level='O0')
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
init()
rank = get_rank()
size = get_group_size()
if size % 2 != 0:
    raise RuntimeError("Group size should be divided by 2 exactly.")


def test_hccl_all_to_all_v_c():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([size, 1]).astype(np.float32)) * rank
    output_tensor = ms.Tensor(np.zeros([size, 1]).astype(np.float32))
    send_count_matrix_size = np.ones([size * size]).astype(np.int64)
    send_count_matrix_size = send_count_matrix_size.tolist()
    handle = all_to_all_v_c(output_tensor, input_tensor, send_count_matrix_size)
    assert handle is None
    except_output_tensor = ms.Tensor(
        np.arange(size).reshape([size, 1]).astype(np.float32)
    )
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # 异步场景
    output_tensor = ms.Tensor(np.zeros([size, 1]).astype(np.float32))
    handle = all_to_all_v_c(output_tensor, input_tensor, send_count_matrix_size, async_op=True)
    assert handle is not None
    handle.wait()
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        create_group(group, list(range(2)))
        if rank == 0:
            tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
            output = ms.Tensor(np.zeros([4, 3]).astype(np.float32))
            handle = all_to_all_v_c(output, tensor, [2, 1, 2, 2], group=group)
            assert handle is None
            except_output_tensor = ms.Tensor(
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
            )
            assert np.allclose(output.asnumpy(), except_output_tensor.asnumpy())
        if rank == 1:
            tensor = ms.Tensor(np.ones([4, 3]).astype(np.float32)) * 2
            output = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
            handle = all_to_all_v_c(output, tensor, [2, 1, 2, 2], group=group)
            assert handle is None
            except_output_tensor = ms.Tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
            assert np.allclose(output.asnumpy(), except_output_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        all_to_all_v_c(1, input_tensor)
    with pytest.raises(TypeError):
        all_to_all_v_c(output_tensor, 1)
    with pytest.raises(TypeError):
        all_to_all_v_c(output_tensor, input_tensor)
    with pytest.raises(TypeError):
        all_to_all_v_c(output_tensor, input_tensor, 1)
    with pytest.raises(TypeError):
        all_to_all_v_c(output_tensor, input_tensor, send_count_matrix_size, group=1)
    with pytest.raises(TypeError):
        all_to_all_v_c(output_tensor, input_tensor, send_count_matrix_size, async_op="1")
    with pytest.raises(TypeError):
        float_array = np.array(send_count_matrix_size, dtype=float)
        all_to_all_v_c(output_tensor, input_tensor, float_array.tolist())


def test_hccl_all_to_all_v_c_001():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([size, 100]).astype(np.int64)) * rank
    output_tensor = ms.Tensor(np.zeros([size, 100]).astype(np.int64))
    send_count_matrix_size = np.ones([size * size]).astype(np.int64)
    send_count_matrix_size = send_count_matrix_size.tolist()
    handle = all_to_all_v_c(output_tensor, input_tensor, send_count_matrix_size)
    assert handle is None
    except_output_tensor = ms.Tensor(
        np.repeat(np.arange(size), 100).reshape([size, 100]).astype(np.int64)
    )
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())


def test_hccl_all_to_all_v_c_002():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([size, 100]).astype(np.uint8)) * rank
    output_tensor = ms.Tensor(np.zeros([size, 100]).astype(np.uint8))
    send_count_matrix_size = np.ones([size * size]).astype(np.int64)
    send_count_matrix_size = send_count_matrix_size.tolist()
    handle = all_to_all_v_c(output_tensor, input_tensor, send_count_matrix_size)
    assert handle is None
    except_output_tensor = ms.Tensor(
        np.repeat(np.arange(size), 100).reshape([size, 100]).astype(np.uint8)
    )
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())


def test_hccl_all_to_all_v_c_003():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # group场景
    if rank == 0 or rank == 1:
        group = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        create_group(group, list(range(2)))
        if rank == 0:
            tensor = ms.Tensor(np.ones([3]).astype(np.float32))
            output = ms.Tensor(shape=(0), dtype=ms.float32)
            handle = all_to_all_v_c(output, tensor, [0, 3, 0, 4], group=group)
            assert handle is None
        if rank == 1:
            tensor = ms.Tensor(np.ones([4]).astype(np.float32)) * 2
            output = ms.Tensor(np.zeros([7]).astype(np.float32))
            handle = all_to_all_v_c(output, tensor, [0, 3, 0, 4], group=group)
            assert handle is None
            except_output_tensor = ms.Tensor([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]])
            assert np.allclose(output.asnumpy(), except_output_tensor.asnumpy())


def test_hccl_all_to_all_v_c_004():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # group场景
    if rank == 0 or rank == 1:
        group = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        create_group(group, list(range(2)))
        if rank == 0:
            tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
            output = ms.Tensor(shape=(0, 3), dtype=ms.float32)
            handle = all_to_all_v_c(output, tensor, [0, 3, 0, 4], group=group)
            assert handle is None
        if rank == 1:
            tensor = ms.Tensor(np.ones([4, 3]).astype(np.float32)) * 2
            output = ms.Tensor(np.zeros([7, 3]).astype(np.float32))
            handle = all_to_all_v_c(output, tensor, [0, 3, 0, 4], group=group)
            assert handle is None
            except_output_tensor = ms.Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                                              [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0],
                                              [2.0, 2.0, 2.0]])
            assert np.allclose(output.asnumpy(), except_output_tensor.asnumpy())


def test_hccl_all_to_all_v_c_005():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # group场景
    input_tensor = ms.Tensor(np.ones([size, 100]).astype(np.int64)) * rank
    output_tensor = ms.Tensor(np.zeros([size, 200]).astype(np.int64))
    send_count_matrix_size = np.ones([size * size]).astype(np.int64)
    send_count_matrix_size = send_count_matrix_size.tolist()
    with pytest.raises(ValueError):
        all_to_all_v_c(output_tensor, input_tensor, send_count_matrix_size)
        _pynative_executor.sync()


def test_hccl_all_to_all_v_c_006():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # group场景
    input_tensor = ms.Tensor(np.ones([size, 100]).astype(np.int64)) * rank
    output_tensor = ms.Tensor(np.zeros([size * 2, 100]).astype(np.int64))
    send_count_matrix_size = np.ones([size * size]).astype(np.int64)
    send_count_matrix_size = send_count_matrix_size.tolist()
    with pytest.raises(RuntimeError):
        all_to_all_v_c(output_tensor, input_tensor, send_count_matrix_size)
        _pynative_executor.sync()
