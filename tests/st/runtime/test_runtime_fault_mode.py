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

import pytest
from mindspore import context, nn, Tensor, ops, jit
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def construct(self, x, y, dim):
        return ops.grad(ops.gather_d)(x, dim, y)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_all_gather_matmul_forward():
    '''
    Feature: Runtime fault test case.
    Description: Sync stream error when invalid input.
    Expectation: Run success
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": 'O2'})
    x = ops.rand(1, 1)
    y = Tensor([[2]])
    with pytest.raises(RuntimeError) as err:
        Net()(x, y, 0)
        _pynative_executor.sync()
    assert (("SyncStream failed" in str(err.value)) or ("Sync Stream failed" in str(err.value))
            or ("Sync stream failed" in str(err.value)))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_max_index_for_gather():
    '''
    Feature: Runtime fault test case.
    Description: Sync stream error when read the illegal memory.
    Expectation: Run success
    '''
    @jit
    def foo(x, y, z):
        x = x - y
        z1 = y / x
        z2 = ops.cast(z1, y.dtype)
        return ops.gather(z, z2, 0)

    x = Tensor(2)
    y = Tensor(2)
    z = Tensor([1, 2, 3, 4])
    with pytest.raises(RuntimeError) as err:
        print(foo(x, y, z))
    assert "Sync" in str(err.value)
