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
    def construct(self, x, dim):
        return x.sort(dim=dim)


class Net1(nn.Cell):
    def construct(self, x):
        return x.sort()


class Net2(nn.Cell):
    def construct(self, x, dim, descending):
        return x.sort(descending=descending, dim=dim)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sort_pyboost(mode):
    """
    Feature: tensor.sort
    Description: Verify the result of sort in pyboost
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), ms.float16)
    net = Net()
    # For the time being, cpu or gpu is not ok in graph mode.
    if ms.get_context('device_target') != 'Ascend' and ms.get_context('mode') == ms.GRAPH_MODE:
        with pytest.raises(RuntimeError):
            net(x, -1)
            _pynative_executor.sync()
        return
    net2 = Net2()
    output1, output2 = net(x, -1)
    output3, output4 = net2(x, -1, False)
    except_out1 = np.array([[1.0000e+00, 2.0000e+00, 8.0000e+00],
                            [3.0000e+00, 5.0000e+00, 9.0000e+00],
                            [4.0000e+00, 6.0000e+00, 7.0000e+00]])
    except_out2 = np.array([[2, 1, 0],
                            [2, 0, 1],
                            [0, 1, 2]])
    np.allclose(output1.asnumpy(), except_out1)
    np.allclose(output2.asnumpy(), except_out2)
    np.allclose(output3.asnumpy(), except_out1)
    np.allclose(output4.asnumpy(), except_out2)


class Net3(nn.Cell):
    def construct(self, x, axis):
        return x.sort(axis=axis)


class Net4(nn.Cell):
    def construct(self, x, axis, descending):
        return x.sort(descending=descending, axis=axis)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sort_python(mode):
    """
    Feature: tensor.sort
    Description: Verify the result of sort in python
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net1 = Net1()
    net3 = Net3()
    net4 = Net4()
    x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), ms.float16)
    output1, output2 = net1(x)
    output3, output4 = net3(x, -1)
    output5, output6 = net4(x, -1, False)
    except_out1 = np.array([[1.0000e+00, 2.0000e+00, 8.0000e+00],
                            [3.0000e+00, 5.0000e+00, 9.0000e+00],
                            [4.0000e+00, 6.0000e+00, 7.0000e+00]])
    except_out2 = np.array([[2, 1, 0],
                            [2, 0, 1],
                            [0, 1, 2]])
    np.allclose(output1.asnumpy(), except_out1)
    np.allclose(output2.asnumpy(), except_out2)
    np.allclose(output3.asnumpy(), except_out1)
    np.allclose(output4.asnumpy(), except_out2)
    np.allclose(output5.asnumpy(), except_out1)
    np.allclose(output6.asnumpy(), except_out2)
