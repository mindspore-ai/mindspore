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
""" test tensor instantiation in pijit """
import os
import pytest
import mindspore as ms
from mindspore import Tensor, jit
from ..share.utils import assert_executed_by_graph_mode
from tests.mark_utils import arg_mark
from tests.st.pi_jit.conftest import run_in_subprocess


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_tensor_instantiation_1():
    """
    Feature: Dynamic tensor instantiation.
    Description: Transform tensor instantiation to primitive. Only compile once
    Expectation: The result is match. Only compile once
    """
    @jit(capture_mode='bytecode')
    def func(x: Tensor):
        return ms.tensor(x), ms.tensor(x, x.dtype)

    for i in range(1, 4):
        data = Tensor([i])
        a, b = func(data)
        assert a == b == data
        assert_executed_by_graph_mode(func, call_count=i)


@run_in_subprocess({'GLOG_v': '1', 'MS_SUBMODULE_LOG_v': '{PI:0}'})
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('data', [1, 1.5, True, [1, 2], [1.5, 2.5], [True, False],
        pytest.param([1, 1.5], marks=pytest.mark.skip(reason="mix types for primitive ListToTensor is unsupported")),
        pytest.param([True, 1.5], marks=pytest.mark.skip(reason="mix types for primitive ListToTensor is unsupported"))
    ])
def test_tensor_instantiation_2(data):
    """
    Feature: Dynamic tensor instantiation.
    Description: Dynamic tensor instantiation.
    Expectation: The result is match.
    """
    @jit(capture_mode='bytecode')
    def func(x, y: Tensor):
        x = ms.tensor(x)
        if x.dtype == ms.bool_:
            x = x.astype(ms.int32)
        return y + x

    os.environ['GLOG_v'] = '0'
    y = Tensor(data)
    excepted = func.__wrapped__(data, y)
    res = func(data, y)
    assert (res == excepted).all()
    # now, scalar input is constant
    assert_executed_by_graph_mode(func)
    del os.environ['GLOG_v']


@run_in_subprocess({'GLOG_v': '1', 'MS_SUBMODULE_LOG_v': '{PI:0}'})
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_tensor_instantiation_3():
    """
    Feature: Tensor instantiation.
    Description: Tensor instantiation.
    Expectation: The result is match.
    """
    @jit(capture_mode='bytecode')
    def func(x, y, z, dtype):
        # now it's constant
        return x + ms.tensor(shape=(x, y, z), dtype=dtype, init=None)

    x=1
    y=2
    z=3
    dtype = ms.int32
    excepted = func.__wrapped__(x, y, z, dtype)
    res = func(x, y, z, dtype)
    assert (res == excepted).all()
    assert_executed_by_graph_mode(func)
