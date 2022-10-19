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
"""multitype_ops directory test case"""
from functools import partial, reduce
import numpy as np

import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import functional as F, composite as C
from ..ut_filter import non_graph_engine
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config


class ScalarAddScalar(nn.Cell):
    """ ScalarAddScalar definition """

    def __init__(self,):
        super(ScalarAddScalar, self).__init__()
        self.n1 = 1.2
        self.n2 = 1.3

    def construct(self):
        return self.n1 + self.n2


class ScalarAddTensor1(nn.Cell):
    """ ScalarAddTensor1 definition """

    def __init__(self,):
        super(ScalarAddTensor1, self).__init__()
        self.t1 = Tensor(np.ones([2, 1, 2, 2], np.float32))
        self.n1 = 1.2
        self.n2 = 1.3

    def construct(self):
        return self.n1 + self.t1


class ScalarAddTensor2(nn.Cell):
    """ ScalarAddTensor2 definition """

    def __init__(self,):
        super(ScalarAddTensor2, self).__init__()
        self.t1 = Tensor(np.ones([2, 1, 2, 2], np.float32))
        self.n1 = 1.2
        self.n2 = 1.3

    def construct(self):
        return self.n1 + self.n2 + self.t1


class TensorAddScalar(nn.Cell):
    """ TensorAddScalar definition """

    def __init__(self,):
        super(TensorAddScalar, self).__init__()
        self.t1 = Tensor(np.ones([2, 1, 2, 2], np.float32))
        self.n1 = 1.2

    def construct(self):
        return self.t1 + self.n1


class ScalarTensorSub(nn.Cell):
    """ ScalarTensorSub definition """

    def __init__(self,):
        super(ScalarTensorSub, self).__init__()
        self.t1 = Tensor(np.ones([2, 1, 2, 2], np.float32))
        self.n1 = 2.1

    def construct(self):
        # scalar - tensor
        z = self.n1 - self.t1
        # tensor - scalar
        z = z - self.n1
        return z


class ScalarTensorMul(nn.Cell):
    """ ScalarTensorMul definition """

    def __init__(self,):
        super(ScalarTensorMul, self).__init__()
        self.t1 = Tensor(np.ones([2, 1, 2, 2], np.float32))
        self.n1 = 2.1

    def construct(self):
        # scalar - tensor
        z = self.n1 * self.t1
        # tensor - scalar
        z = z * self.n1
        return z


class ScalarTensorDiv(nn.Cell):
    """ ScalarTensorDiv definition """

    def __init__(self,):
        super(ScalarTensorDiv, self).__init__()
        self.t1 = Tensor(np.ones([2, 1, 2, 2], np.float32))
        self.n1 = 2.1

    def construct(self):
        # scalar - tensor
        z = self.n1 / self.t1
        # tensor - scalar
        z = z / self.n1
        return z


class EqualClass(nn.Cell):
    def __init__(self, x, y):
        super(EqualClass, self).__init__()
        self.n1 = x
        self.n2 = y

    def construct(self):
        if self.n1 == self.n2:
            return self.n1
        return self.n2


grad_scale = C.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Number", "Tensor")
def tensor_grad_scale(scale, grad):
    """Get grad with scale."""
    if scale == 1.0:
        return grad
    return grad * F.scalar_to_tensor(scale)


class MapPartialNet(nn.Cell):
    def __init__(self):
        super(MapPartialNet, self).__init__()
        self.reciprocal_scale = 1.2
        self.x1 = Tensor(np.ones([2, 1, 2,], np.float32))
        self.x2 = Tensor(np.ones([2, 1, 2, 2], np.float32))

    def construct(self, x, y):
        grads = (self.x1, self.x2, x, y)
        grads = map(partial(grad_scale, self.reciprocal_scale), grads)
        return grads


class ZipNet(nn.Cell):
    def __init__(self):
        super(ZipNet, self).__init__()
        self.x1 = Tensor(np.ones([1, 2, 2, 1], np.float32))
        self.x2 = Tensor(np.ones([2, 1, 2, 2], np.float32))

    def construct(self, x, y):
        t1 = (self.x1, self.x2, x, y)
        t2 = (y, x, self.x2, self.x1)
        t3 = zip(t1, t2)
        return t3


class UnZipNet(nn.Cell):
    def __init__(self):
        super(UnZipNet, self).__init__()
        self.x1 = Tensor(np.ones([1, 2, 2, 1], np.float32))
        self.x2 = Tensor(np.ones([2, 1, 2, 2], np.float32))

    def construct(self, x, y):
        t1 = (self.x1, self.x2, x, y)
        t2 = (y, x, self.x2, self.x1)
        t3 = zip(t1, t2)
        t4 = zip(*t3)
        return t4


class ScalarTensorOp2Cast(nn.Cell):
    def __init__(self,):
        super(ScalarTensorOp2Cast, self).__init__()
        self.f = 1.2
        self.t = Tensor(np.ones([2, 1, 2, 2], np.float16))

    def construct(self):
        a1 = self.f + self.t
        a2 = self.t + self.f
        a = a1 + a2
        b1 = self.f - self.t
        b2 = self.t - self.f
        b = b1 - b2
        c1 = self.f * self.t
        c2 = self.t * self.f
        c = c1 * c2
        d1 = self.t / self.f
        d2 = self.f / self.t
        d = d1 / d2
        x = a + b
        y = c + d
        z = x + y
        return z


test_case_ops = [
    ('ScalarAddScalar', {
        'block': ScalarAddScalar(),
        'desc_inputs': []}),
    ('ScalarAddTensor1', {
        'block': ScalarAddTensor1(),
        'desc_inputs': []}),
    ('ScalarAddTensor2', {
        'block': ScalarAddTensor2(),
        'desc_inputs': []}),
    ('TensorAddScalar', {
        'block': TensorAddScalar(),
        'desc_inputs': []}),
    ('ScalarTensorSub', {
        'block': ScalarTensorSub(),
        'desc_inputs': []}),
    ('ScalarTensorMul', {
        'block': ScalarTensorMul(),
        'desc_inputs': []}),
    ('ScalarTensorDiv', {
        'block': ScalarTensorDiv(),
        'desc_inputs': []}),
    ('ScalarEqScalar', {
        'block': EqualClass(1, 2),
        'desc_inputs': []}),
    ('ScalarEqNone', {
        'block': EqualClass(1, None),
        'desc_inputs': []}),
    ('NoneEqScalar', {
        'block': EqualClass(None, 2),
        'desc_inputs': []}),
    ('TupleEqTuple', {
        'block': EqualClass((1, 2), (1, 3)),
        'desc_inputs': []}),
    ('NoneEqTuple', {
        'block': EqualClass(None, (1, 3)),
        'desc_inputs': []}),
    ('TupleEqNone', {
        'block': EqualClass((1, 2), None),
        'desc_inputs': []}),
    ('EqTensor1', {
        'block': EqualClass(Tensor(np.ones([2, 1, 2, 2], np.float32)),
                            Tensor(np.ones([2, 1, 2, 2], np.float32))),
        'desc_inputs': []}),
    ('EqTensor2', {
        'block': EqualClass(Tensor(np.ones([2, 1, 2, 2], np.float32)), None),
        'desc_inputs': []}),
    ('NoneEqTensor', {
        'block': EqualClass(None, Tensor(np.ones([2, 1, 2, 2], np.float32))),
        'desc_inputs': []}),
    ('MapPartial', {
        'block': MapPartialNet(),
        'desc_inputs': [Tensor(np.ones([2, 1, 2, 2], np.float32)),
                        Tensor(np.ones([2, 1, 2, 2], np.float32))]}),
    ('Zip', {
        'block': ZipNet(),
        'desc_inputs': [Tensor(np.ones([2, 1, 2, 2], np.float32)),
                        Tensor(np.ones([2, 1, 2, 2], np.float32))]}),
    ('Unzip', {
        'block': UnZipNet(),
        'desc_inputs': [Tensor(np.ones([2, 1, 2, 2], np.float32)),
                        Tensor(np.ones([2, 1, 2, 2], np.float32))]}),
    ('ScalarTensorOpCast2', {
        'block': ScalarTensorOp2Cast(),
        'desc_inputs': []}),
]

test_case_lists = [test_case_ops]
test_exec_case = reduce(lambda x, y: x + y, test_case_lists)
# use -k to select certain testcast
# pytest tests/python/ops/test_ops.py::test_backward -k LayerNorm



@non_graph_engine
@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_exec_case
