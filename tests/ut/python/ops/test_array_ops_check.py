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
""" test ops """
import numpy as np

import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config_exception

context.set_context(mode=context.PYNATIVE_MODE)


class ExpandDimsNet(nn.Cell):
    def __init__(self, axis):
        super(ExpandDimsNet, self).__init__()
        self.axis = axis
        self.op = P.ExpandDims()

    def construct(self, x):
        return self.op(x, self.axis)



class ReshapeNet(nn.Cell):
    def __init__(self, shape):
        super(ReshapeNet, self).__init__()
        self.shape = shape
        self.op = P.Reshape()

    def construct(self, x):
        return self.op(x, self.shape)


raise_set = [
    # input is scala, not Tensor
    ('ExpandDims0', {
        'block': (P.ExpandDims(), {'exception': TypeError, 'error_keywords': ['ExpandDims']}),
        'desc_inputs': [5.0, 1],
        'skip': ['backward']}),
    # axis is as a parameter
    ('ExpandDims1', {
        'block': (P.ExpandDims(), {'exception': TypeError, 'error_keywords': ['ExpandDims']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32)), 1],
        'skip': ['backward']}),
    # axis as an attribute, but less then lower limit
    ('ExpandDims2', {
        'block': (ExpandDimsNet(-4), {'exception': ValueError, 'error_keywords': ['ExpandDims']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # axis as an attribute, but greater then upper limit
    ('ExpandDims3', {
        'block': (ExpandDimsNet(3), {'exception': ValueError, 'error_keywords': ['ExpandDims']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),

    # input is scala, not Tensor
    ('DType0', {
        'block': (P.DType(), {'exception': TypeError, 'error_keywords': ['DType']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),

    # input x scala, not Tensor
    ('SameTypeShape0', {
        'block': (inner.SameTypeShape(), {'exception': TypeError, 'error_keywords': ['SameTypeShape']}),
        'desc_inputs': [5.0, Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # input y scala, not Tensor
    ('SameTypeShape1', {
        'block': (inner.SameTypeShape(), {'exception': TypeError, 'error_keywords': ['SameTypeShape']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32)), 5.0],
        'skip': ['backward']}),
    # type of x and y not match
    ('SameTypeShape2', {
        'block': (inner.SameTypeShape(), {'exception': TypeError, 'error_keywords': ['SameTypeShape']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32)), Tensor(np.ones([3, 4]).astype(np.int32))],
        'skip': ['backward']}),
    # shape of x and y not match
    ('SameTypeShape3', {
        'block': (inner.SameTypeShape(), {'exception': ValueError, 'error_keywords': ['SameTypeShape']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32)), Tensor(np.ones([3, 3]).astype(np.float32))],
        'skip': ['backward']}),

    # sub_type is None
    ('IsSubClass0', {
        'block': (inner.IsSubClass(), {'exception': TypeError, 'error_keywords': ['IsSubClass']}),
        'desc_inputs': [None, mstype.number],
        'skip': ['backward']}),
    # type_ is None
    ('IsSubClass1', {
        'block': (inner.IsSubClass(), {'exception': TypeError, 'error_keywords': ['IsSubClass']}),
        'desc_inputs': [mstype.number, None],
        'skip': ['backward']}),

    # input x is scalar, not Tensor
    ('Reshape0', {
        'block': (P.Reshape(), {'exception': TypeError, 'error_keywords': ['Reshape']}),
        'desc_inputs': [5.0, (1, 2)],
        'skip': ['backward']}),
    # input shape is var
    ('Reshape1', {
        'block': (P.Reshape(), {'exception': TypeError, 'error_keywords': ['Reshape']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32)), (2, 3, 2)],
        'skip': ['backward']}),
    # element of shape is not int
    ('Reshape3', {
        'block': (ReshapeNet((2, 3.0, 2)), {'exception': TypeError, 'error_keywords': ['Reshape']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
]


@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config_exception)
def test_check_exception():
    return raise_set
