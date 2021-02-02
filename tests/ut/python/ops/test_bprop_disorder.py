# Copyright 2019 Huawei Technologies Co., Ltd
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
""" test bprop disorder """
import functools
import numpy as np

import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor, Parameter
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from ..ut_filter import non_graph_engine
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config


grad_by_list_with_sens = C.GradOperation(get_by_list=True, sens_param=True)


class DisOrderTest1(nn.Cell):
    """ DisOrderTest1 definition """

    def __init__(self):
        super(DisOrderTest1, self).__init__()
        weight = Tensor(np.ones([1], np.float32))
        self.s1 = Parameter(weight, name="s1")
        self.s2 = Parameter(weight, name="s2")
        self.s3 = Parameter(weight, name="s3")
        self.s4 = Parameter(weight, name="s4")
        self.mul = P.Mul()
        self.add = P.Add()

    def construct(self, x):
        return x * (self.s1 * self.s2 + self.s2 * self.s3 + self.s3 * self.s4 + self.s4 * self.s1)


class DisOrderTest2(nn.Cell):
    """ DisOrderTest2 definition """

    def __init__(self):
        super(DisOrderTest2, self).__init__()
        weight = Tensor(np.ones([1], np.float32))
        self.s1 = Parameter(weight, name="s1")
        self.s2 = Parameter(weight, name="s2")
        self.s3 = Parameter(weight, name="s3")
        self.s4 = Parameter(weight, name="s4")
        self.mul = P.Mul()
        self.add = P.Add()

    def construct(self, x):
        return self.mul(x, (self.add(self.add(self.add(self.mul(self.s1, self.s2), self.mul(self.s2, self.s3)),
                                              self.mul(self.s3, self.s4)), self.mul(self.s4, self.s1))))


class GradNetWrap(nn.Cell):
    """ GradNetWrap definition """

    def __init__(self, net):
        super(GradNetWrap, self).__init__()
        self.net = net
        self.weights = ParameterTuple(net.get_parameters())

    def construct(self, x, sens):
        return grad_by_list_with_sens(self.net, self.weights)(x, sens)


test_case_ops = [
    ('DisOrderTest1', {
        'block': GradNetWrap(DisOrderTest1()),
        'desc_inputs': [Tensor(np.ones([1], np.float32)), Tensor(np.ones([1], np.float32))]}),
    ('DisOrderTest2', {
        'block': GradNetWrap(DisOrderTest2()),
        'desc_inputs': [Tensor(np.ones([1], np.float32)), Tensor(np.ones([1], np.float32))]}),
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
