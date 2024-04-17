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
import os
import numpy as np
from mindspore import Tensor, ops, nn
import mindspore.context as context


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def construct(self, a, b):
        out = ops.add(a, b)
        return out


def test_ms_format_mode_0():
    """
    Feature: for MS_FORMAT_MODE is 0
    Description: net is add net
    Expectation: the result is correct
    """
    os.environ['MS_FORMAT_MODE'] = '0'
    net = Net()
    x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    y = Tensor(np.array([4, 5, 6]).astype(np.float32))
    out = net(x, y)
    expect = np.array([5, 7, 9]).astype(np.float32)
    assert (out.asnumpy() == expect).all
    del os.environ['MS_FORMAT_MODE']

def test_ms_format_mode_1():
    """
    Feature: for MS_FORMAT_MODE is 1
    Description: net is add net
    Expectation: the result is correct
    """
    os.environ['MS_FORMAT_MODE'] = '1'
    net = Net()
    x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    y = Tensor(np.array([4, 5, 6]).astype(np.float32))
    out = net(x, y)
    expect = np.array([5, 7, 9]).astype(np.float32)
    assert (out.asnumpy() == expect).all
    del os.environ['MS_FORMAT_MODE']
