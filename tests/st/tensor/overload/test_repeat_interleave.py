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
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def construct(self, x, repeats, dim):
        return x.repeat_interleave(repeats, dim)


class Net1(nn.Cell):
    def construct(self, x, repeats):
        return x.repeat_interleave(repeats)


class Net2(nn.Cell):
    def construct(self, x, repeats, dim):
        return x.repeat_interleave(dim=dim, repeats=repeats)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_repeat_interleave_python(mode):
    """
    Feature: Tensor.repeat_interleave
    Description: Verify the result of repeat_interleave in python
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    net1 = Net1()
    net2 = Net2()
    input_x = Tensor(np.array([1, 2, 3]), ms.int32)
    # test diff args when repeats type is int
    output1 = np.array([1, 1, 2, 2, 3, 3])
    assert np.allclose(net1(input_x, 2).asnumpy(), output1)
    assert np.allclose(net(input_x, 2, None).asnumpy(), output1)
    assert np.allclose(net2(input_x, 2, None).asnumpy(), output1)

    input_y = Tensor(np.array([[1, 2], [3, 4]]), ms.int32)
    output2 = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    assert np.allclose(net1(input_y, 2).asnumpy(), output2)

    output3 = np.array([[1, 1, 1, 2, 2, 2],
                        [3, 3, 3, 4, 4, 4]])
    assert np.allclose(net(input_y, 3, 1).asnumpy(), output3)

    # test diff args when repeats type is Tensor
    output4 = np.array([[1, 2],
                        [3, 4],
                        [3, 4]])
    if ms.get_context('mode') == ms.GRAPH_MODE:
        with pytest.raises(RuntimeError):
            net2(input_y, Tensor(np.array([1, 2])), 0)
        return
    assert np.allclose(net2(input_y, Tensor(np.array([1, 2])), 0).asnumpy(), output4)


class Net3(nn.Cell):
    def construct(self, x, repeats, dim, output_size):
        return x.repeat_interleave(repeats, dim, output_size)


class Net4(nn.Cell):
    def construct(self, x, repeats, dim, output_size):
        return x.repeat_interleave(repeats, dim, output_size=output_size)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu',
                      'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_repeat_interleave_pyboost_fault(mode):
    """
    Feature: Tensor.repeat_interleave
    Description: Verify the result of repeat_interleave in pyboost which raises fault
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net4 = Net4()
    input_x = Tensor(np.array([1, 2, 3]), ms.int32)
    # aclnn oprrator only supports ascend 910b
    if ms.get_context('device_target') != 'platform_ascend910b':
        with pytest.raises(RuntimeError):
            net4(input_x, 2, None, None)
            _pynative_executor.sync()
        return


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_repeat_interleave_pyboost_ok(mode):
    """
    Feature: Tensor.repeat_interleave
    Description: Verify the result of repeat_interleave in pyboost
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net3 = Net3()
    net4 = Net4()
    input_x = Tensor(np.array([1, 2, 3]), ms.int32)
    # test kwonlyargs in pynative mode
    if ms.get_context('mode') == ms.PYNATIVE_MODE:
        with pytest.raises(TypeError):
            net3(input_x, 2, None, None)
            _pynative_executor.sync()

    output1 = np.array([1, 1, 2, 2, 3, 3])
    assert np.allclose(net4(input_x, 2, None, None).asnumpy(), output1)
    input_y = Tensor(np.array([[1, 2], [3, 4]]), ms.int32)
    output2 = np.array([[1, 2],
                        [3, 4],
                        [3, 4]])
    assert np.allclose(net4(input_y, Tensor(np.array([1, 2])), 0, None).asnumpy(), output2)
