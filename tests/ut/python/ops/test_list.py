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
import functools
import numpy as np

import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P
from ..ut_filter import non_graph_engine
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config


def test_list_equal():
    class Net(nn.Cell):
        def __init__(self, z: list):
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z == [1, 2, 3]:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = [1, 2, 3]
    net = Net(z)
    assert net(x, y) == x


def test_list_not_equal():
    class Net(nn.Cell):
        def __init__(self, z: list):
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z == [3, 4, 5]:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = [1, 2, 3]
    net = Net(z)
    assert net(x, y) == y


def test_list_expansion():
    class Net(nn.Cell):
        def __init__(self, z: list):
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            a, b, c = self.z
            if a == 1 and b == 2 and c == 3:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = [1, 2, 3]
    net = Net(z)
    assert net(x, y) == x


def test_list_append():
    class Net(nn.Cell):
        def __init__(self, z: list):
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            z = [[1, 2], 3]
            z[0].append(88)
            z[0].append(99)
            if z[0][3] == 99:
                ret = y
            else:
                ret = x
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = [1, 2, 3]
    net = Net(z)
    assert net(x, y) == y


def test_list_append_2():
    class Net(nn.Cell):
        def __init__(self, z: list):
            super(Net, self).__init__()
            self.z = z
            self.x = 9

        def construct(self, x, y):
            self.z[0].append(88)
            self.z[0].append(99)
            if self.z[0][3] == 88:
                ret = y
            else:
                ret = x
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = [[1, 2], 3]
    net = Net(z)
    assert net(x, y) == x


class ListOperate(nn.Cell):
    def __init__(self,):
        super(ListOperate, self).__init__()

    def construct(self, t, l):
        x = [1, 2, 3, 4, 5, 6]
        x[2] = 9
        x[1] = x[3] + 11
        x[3] = x[1] + x[0]
        x[0] = x[2] * x[4]
        x[5] = x[1] - x[2]
        x[4] = x[3] / x[2]
        x.append(8)
        x.append(8)
        x.append(t)
        x.append(l)
        x.append(l)
        return x


class AxisListNet(nn.Cell):
    def __init__(self):
        super(AxisListNet, self).__init__()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reduce_max = P.ReduceMax()
        self.reduce_min = P.ReduceMin()
        self.add_n = P.AddN()
        self.axis = [0, 1, 2]

    def construct(self, x):
        ret_sum = self.reduce_sum(x, self.axis)
        ret_mean = self.reduce_mean(x, self.axis)
        ret_max = self.reduce_max(x, self.axis)
        ret_min = self.reduce_min(x, self.axis)
        ret = [ret_sum, ret_mean, ret_max, ret_min]
        return self.add_n(ret) + ret_sum


class AxisListEmptyNet(nn.Cell):
    def __init__(self):
        super(AxisListEmptyNet, self).__init__()
        self.reduce_sum = P.ReduceSum()
        self.axis = []

    def construct(self, x):
        return self.reduce_sum(x, self.axis)


class AxisListDefaultNet(nn.Cell):
    def __init__(self):
        super(AxisListDefaultNet, self).__init__()
        self.reduce_sum = P.ReduceSum()

    def construct(self, x):
        return self.reduce_sum(x)


test_case_ops = [
    ('ListOperate', {
        'block': ListOperate(),
        'desc_inputs': [Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32)),
                        [2, 3, 4]]}),
    ('AxisList', {
        'block': AxisListNet(),
        'desc_inputs': [Tensor(np.ones([6, 8, 10], np.int32))]}),
    ('AxisListEmpty', {
        'block': AxisListEmptyNet(),
        'desc_inputs': [Tensor(np.ones([6, 8, 10], np.int32))]}),
    ('AxisListDefault', {
        'block': AxisListDefaultNet(),
        'desc_inputs': [Tensor(np.ones([6, 8, 10], np.int32))]}),
]

test_case_lists = [test_case_ops]
test_exec_case = functools.reduce(lambda x, y: x + y, test_case_lists)
# use -k to select certain testcast
# pytest tests/python/ops/test_ops.py::test_backward -k LayerNorm


@non_graph_engine
@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_exec_case
