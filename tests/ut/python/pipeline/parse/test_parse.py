# Copyright 2020 Huawei Technologies Co., Ltd
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
@File  : test_parse.py
@Author:
@Date  : 2019-01-23 17:13
@Desc  :
"""
import logging
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import ms_function, _executor
from mindspore.ops.composite import core
from mindspore.ops.functional import tensor_add
from ...ut_filter import non_graph_engine
# pylint: disable=W0613
# W0613: unused-argument


log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)

# Test case: use the parse obj interface use default parameter
class Net(nn.Cell):
    """ Net definition """
    def __init__(self, dim):
        super(Net, self).__init__()
        self.softmax1 = nn.Softmax(dim)
        self.softmax2 = nn.Softmax(dim + 1)

    def construct(self, input_data, input1=ms.Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))):
        return self.softmax1(input_data)


@non_graph_engine
def test_parse_defalut_parameter_case2():
    """ test_parse_defalut_parameter_case2 """
    log.debug("begin test_parse_defalut_parameter_case2")
    net = Net(0)
    npd = np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32')
    log.debug("input value is: %r", npd)
    input_data = ms.Tensor(npd)
    input_data.set_dtype(ms.float32)

    log.debug("start run")
    output = net(input_data)

    value = output.asnumpy()
    log.debug("output value = %r", value)



# Test case: use the variable parameter for parse object
class Net1(nn.Cell):
    """ Net1 definition """
    def __init__(self):
        super(Net1, self).__init__()

    def construct(self, *args):
        x = args[0]
        return x


def test_var_parameter_case2():
    """ test_var_parameter_case2 """
    log.debug("begin test_var_parameter_case2")
    net = Net1()
    npd = np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32')
    log.debug("input value is: %r", npd)
    input_data = ms.Tensor(npd)
    input_data.set_dtype(ms.float32)

    np1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input1 = ms.Tensor(np1)
    np2 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input2 = ms.Tensor(np2)

    _executor.compile(net, input_data, input1, input2)



# Test case: test the global flag
g_x = Tensor(np.ones([3, 3]).astype(np.float32))

@ms_function
def tensor_add_global(x):
    """ tensor_add_global """
    global g_x
    res = tensor_add(x, g_x)
    return res


@non_graph_engine
def test_global_flag():
    """ test_global_flag """
    log.debug("begin test_global_flag")
    x = Tensor(np.ones([3, 3]).astype(np.float32))
    res = tensor_add_global(x)
    log.debug("finished test_global_flag, ret = %r", res)


class NetWithNDarray(nn.Cell):
    """ NetWithNDarray definition """
    def __init__(self, dim):
        super(NetWithNDarray, self).__init__()
        self.softmax = nn.Softmax(dim)
        self.x = ms.Tensor(np.ones(shape=(1)).astype(np.float32))

    def construct(self, input_data):
        return self.softmax(input_data) * self.x

@non_graph_engine
def test_net_with_ndarray():
    """ test_net_with_ndarray """
    net = NetWithNDarray(0)
    input_data = np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32')

    net(ms.Tensor(input_data))
