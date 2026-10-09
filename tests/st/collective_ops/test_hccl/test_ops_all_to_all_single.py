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

"""test hccl AllGather and all_gather with 8p"""

import os
import numpy as np
import pytest
from mindspore import Tensor, runtime
from mindspore import dtype as mstype
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.ops.communication import all_to_all_single, set_comm_ops_inplace
from mindspore.ops.function.comm_func import is_inplace_func
from mindspore import context, jit

np.random.seed(1)
context.set_context(jit_level='O0')
os.environ['HCCL_WHITELIST_DISABLE'] = str(1)
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
init()
#msrun --worker_num=2 --local_worker_num=2 --master_port=1128 --log_dir=msrun_log --join=True --cluster_time_out=300  test_ops_all_to_all_single.py


def test_hccl_all_to_all_single_func_dynstatic1():
    """
    Feature: test 'all_to_all_single' communication function.
    Description: test 'all_to_all_single' communication function.
    Expectation: expect correct result.
    """
    @jit(jit_level="O0")
    def test(out_tensor, in_tensor):
        """
        Feature: test 'all_to_all_single' communication function.
        Description: test 'all_to_all_single' communication function.
        Expectation: expect correct result.
        """
        output, h = all_to_all_single(out_tensor, in_tensor, async_op=True)
        h.wait()
        return output

    @jit(jit_level="O0")
    def test1(out_tensor, in_tensor):
        """
        Feature: test 'all_to_all_single' communication function.
        Description: test 'all_to_all_single' communication function.
        Expectation: expect correct result.
        """
        output, h = all_to_all_single(out_tensor, in_tensor, async_op=True)
        return output, h

    set_comm_ops_inplace(False)
    context.set_context(save_graphs=3, save_graphs_path="ir_dump_path")
    x = np.ones([6, 4]).astype(np.float32)
    expect_output = np.ones([6, 4]).astype(np.float32)
    output = test(None, Tensor(x, mstype.float32))
    assert np.allclose(output.asnumpy(), expect_output)

    output, h = test1(None, Tensor(x, mstype.float32))
    h.wait()
    assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_all_to_all_single_func_dynstatic2():
    """
    Feature: test 'all_to_all_single' communication function.
    Description: test 'all_to_all_single' communication function.
    Expectation: expect correct result.
    """
    @jit(jit_level="O0")
    def test_inplace(out_tensor, in_tensor):
        """
        Feature: test 'all_to_all_single' communication function.
        Description: test 'all_to_all_single' communication function.
        Expectation: expect correct result.
        """
        h = all_to_all_single(out_tensor, in_tensor, async_op=True)
        if isinstance(h, tuple):
            h[1].wait()
        else:
            h.wait()
        return h

    @jit(jit_level="O0")
    def test_inplace1(out_tensor, in_tensor):
        """
        Feature: test 'all_to_all_single' communication function.
        Description: test 'all_to_all_single' communication function.
        Expectation: expect correct result.
        """
        h = all_to_all_single(out_tensor, in_tensor, async_op=True)
        return h

    set_comm_ops_inplace(True)
    context.set_context(save_graphs=3, save_graphs_path="ir_dump_path")
    x = Tensor(np.ones([6, 40]).astype(np.float32), mstype.float32)
    y = Tensor(np.zeros([6, 40]).astype(np.float32), mstype.float32)
    expect_output = np.ones([6, 40]).astype(np.float32)
    h = test_inplace(y, x)
    assert np.allclose(y.asnumpy(), expect_output)
    y = Tensor(np.zeros([6, 40]).astype(np.float32), mstype.float32)
    h = test_inplace1(y, x)
    if isinstance(h, tuple):
        h[1].wait()
    else:
        h.wait()
    assert np.allclose(y.asnumpy(), expect_output)


@pytest.fixture(name="inplace_all_to_all_single")
def enable_inplace_all_to_all_single():
    """Select the in-place all_to_all_single implementation and restore its previous setting."""
    previous = is_inplace_func("all_to_all_single")
    set_comm_ops_inplace(True, ["all_to_all_single"])
    yield
    set_comm_ops_inplace(previous, ["all_to_all_single"])


@pytest.mark.usefixtures("inplace_all_to_all_single")
@pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("splits_by_rank,expected_by_rank", [
    pytest.param(
        (None, None),
        ([1 + 11j, 2 + 12j, 5 + 15j, 6 + 16j], [3 + 13j, 4 + 14j, 7 + 17j, 8 + 18j]),
        id="equal-omitted",
    ),
    pytest.param(
        ([2, 2], [2, 2]),
        ([1 + 11j, 2 + 12j, 5 + 15j, 6 + 16j], [3 + 13j, 4 + 14j, 7 + 17j, 8 + 18j]),
        id="equal-explicit",
    ),
    pytest.param(
        ([1, 3], [3, 1]),
        ([1 + 11j, 5 + 15j, 6 + 16j, 7 + 17j], [2 + 12j, 3 + 13j, 4 + 14j, 8 + 18j]),
        id="uneven-explicit",
    ),
])
def test_complex64_exchange_preserves_all_components_and_peer_offsets(async_op, splits_by_rank, expected_by_rank):
    """
    Feature: ops.communication.all_to_all_single complex64 communication.
    Description: Exchange distinct complex values in place with omitted, equal explicit and uneven explicit splits.
    Expectation: Every real and imaginary component arrives at the correct peer offset.
    """
    assert get_group_size() == 2, "Launch with msrun --worker_num=2 --local_worker_num=2."
    rank = get_rank()
    inputs = ([1 + 11j, 2 + 12j, 3 + 13j, 4 + 14j], [5 + 15j, 6 + 16j, 7 + 17j, 8 + 18j])
    input_tensor = Tensor(inputs[rank], mstype.complex64).move_to("Ascend")
    output = Tensor([-99 - 199j] * 4, mstype.complex64).move_to("Ascend")
    splits = splits_by_rank[rank]
    runtime.synchronize()

    handle = all_to_all_single(output, input_tensor, splits, splits, async_op=async_op)
    if async_op:
        assert handle is not None
        handle.wait()
    else:
        assert handle is None
    runtime.synchronize()

    assert output.dtype == mstype.complex64
    np.testing.assert_array_equal(output.asnumpy(), np.asarray(expected_by_rank[rank], dtype=np.complex64))
