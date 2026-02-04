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
import hashlib
import mindspore as ms
from mindspore import context
from mindspore.ops import cat
from mindspore.ops.communication import (
    init_process_group,
    get_rank,
    get_world_size,
    new_group,
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


def test_hccl_scatter_tensor():
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
    except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_handle = scatter_tensor(output_tensor, input_tensor, src=0)
    assert output_handle is None
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # 异步场景
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    output_handle = scatter_tensor(output_tensor, input_tensor, src=0, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.ones([3 * 2, 3]).astype(np.float32))
        output_tensor1 = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
        output_handle = scatter_tensor(output_tensor1, input_tensor1, src=0, group=name)
        except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())


def test_hccl_gather_into_tensor():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_tensor = ms.Tensor(np.zeros([3 * size, 3]).astype(np.float32))
    output_handle = gather_into_tensor(output_tensor, input_tensor)
    assert output_handle is None
    if rank == 0:
        except_output_tensor = ms.Tensor(np.ones([3 * size, 3]).astype(np.float32))
        assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    else:
        except_output_tensor = ms.Tensor(np.zeros([3 * size, 3]).astype(np.float32))
        assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # 异步场景
    output_tensor = ms.Tensor(np.zeros([3 * size, 3]).astype(np.float32))
    output_handle = gather_into_tensor(output_tensor, input_tensor, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
        output_tensor1 = ms.Tensor(np.zeros([6, 3]).astype(np.float32))
        output_handle = gather_into_tensor(
            output_tensor1, input_tensor1, dst=1, group=name
        )
        assert output_handle is None
        if rank == 1:
            except_output_tensor = cat([input_tensor1, input_tensor1])
            assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())
        else:
            except_output_tensor = ms.Tensor(np.zeros([6, 3]).astype(np.float32))
            assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())
