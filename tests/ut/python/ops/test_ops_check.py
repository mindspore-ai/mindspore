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
"""test checking for some ops"""
import functools
import logging
import numpy as np
import mindspore.context as context
from mindspore import Tensor
from mindspore import nn
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P
from ..ut_filter import non_graph_engine
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config
from ....mindspore_test_framework.pipeline.forward.verify_exception \
    import pipeline_for_verify_exception_for_case_by_case_config

logging.basicConfig(level=logging.WARNING)


# pylint: disable=abstract-method
class NetMissConstruct(nn.Cell):
    """ NetMissConstruct definition """

    def __init__(self):
        super(NetMissConstruct, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.flatten = P.Flatten()

    # TestCase: Mis-spelled 'construct' to 'construtc'
    def construtc(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_net_without_construct():
    """ test_net_without_construct """
    net = NetMissConstruct()
    inp = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
    _cell_graph_executor.compile(net, inp)


class NetAddN(nn.Cell):
    """net for test AddN"""

    def __init__(self):
        super(NetAddN, self).__init__()
        self.net = P.AddN()

    def construct(self, x):
        return self.net(x)


class NetSplit(nn.Cell):
    "net for test Split"

    def __init__(self):
        super(NetSplit, self).__init__()
        self.net = P.Split(1, 2)

    def construct(self, x):
        return self.net(x)


class NetBatchMatMul(nn.Cell):
    """net for test BatchMatMul"""

    def __init__(self):
        super(NetBatchMatMul, self).__init__()
        self.op = P.BatchMatMul()

    def construct(self, x, y):
        return self.op(x, y)


test_case_check_ops = [
    ('Conv_Padding_1', {
        'block': nn.Conv2d(1, 6, 5, pad_mode='same', padding=0),
        'desc_inputs': [Tensor(np.ones(shape=[1, 1, 6, 5]).astype(np.float32))]}),
    ('Conv_Padding_2', {
        'block': nn.Conv2d(1, 6, 5, pad_mode='valid', padding=0),
        'desc_inputs': [Tensor(np.ones(shape=[1, 1, 6, 5]).astype(np.float32))]}),
    ('Conv_Padding_3', {
        'block': nn.Conv2d(1, 6, 5, pad_mode='pad', padding=0),
        'desc_inputs': [Tensor(np.ones(shape=[1, 1, 6, 5]).astype(np.float32))]}),
    ('Conv_Padding_4', {
        'block': nn.Conv2d(1, 6, 5, pad_mode='pad', padding=7),
        'desc_inputs': [Tensor(np.ones(shape=[1, 1, 6, 5]).astype(np.float32))]}),
    ('Conv_Bias_1', {
        'block': nn.Conv2d(1, 6, 5, has_bias=True, bias_init=Tensor(np.ones([6]).astype(np.float32))),
        'desc_inputs': [Tensor(np.ones(shape=[1, 1, 6, 5]).astype(np.float32))]}),
    ('Conv_Bias_2', {
        'block': nn.Conv2d(1, 6, 5, has_bias=True, bias_init='zeros'),
        'desc_inputs': [Tensor(np.ones(shape=[1, 1, 6, 5]).astype(np.float32))]}),
    ('Conv_Bias_3', {
        'block': nn.Conv2d(1, 6, 5, has_bias=False, bias_init='zeros'),
        'desc_inputs': [Tensor(np.ones(shape=[1, 1, 6, 5]).astype(np.float32))]}),
    ('Conv_Bias_4', {
        'block': nn.Conv2d(1, 6, 5, has_bias=False, bias_init=Tensor(np.ones([6]).astype(np.float32))),
        'desc_inputs': [Tensor(np.ones(shape=[1, 1, 6, 5]).astype(np.float32))]}),
    ('Dense_Bias_1', {
        'block': nn.Dense(1, 6, has_bias=True, bias_init=Tensor(np.ones([6]).astype(np.float32))),
        'desc_inputs': [Tensor(np.ones(shape=[6, 1]).astype(np.float32))]}),
    ('Dense_Bias_2', {
        'block': nn.Dense(1, 6, has_bias=True, bias_init='zeros'),
        'desc_inputs': [Tensor(np.ones(shape=[6, 1]).astype(np.float32))]}),
    ('Dense_Bias_3', {
        'block': nn.Dense(1, 6, has_bias=False, bias_init='zeros'),
        'desc_inputs': [Tensor(np.ones(shape=[6, 1]).astype(np.float32))]}),
    ('Dense_Bias_4', {
        'block': nn.Dense(1, 6, has_bias=False, bias_init=Tensor(np.ones([6]).astype(np.float32))),
        'desc_inputs': [Tensor(np.ones(shape=[6, 1]).astype(np.float32))]}),
    ('MaxPool2d_1', {
        'block': nn.MaxPool2d(5, pad_mode='same'),
        'desc_inputs': [Tensor(np.ones(shape=[5, 5, 8, 8]).astype(np.float32))]}),
    ('MaxPool2d_2', {
        'block': nn.MaxPool2d(5, pad_mode='valid'),
        'desc_inputs': [Tensor(np.ones(shape=[5, 5, 8, 8]).astype(np.float32))]}),
    ('AvgPool2d_1', {
        'block': nn.AvgPool2d(5, pad_mode='same'),
        'desc_inputs': [Tensor(np.ones(shape=[5, 5, 8, 8]).astype(np.float32))]}),
    ('AvgPool2d_2', {
        'block': nn.AvgPool2d(5, pad_mode='valid'),
        'desc_inputs': [Tensor(np.ones(shape=[5, 5, 8, 8]).astype(np.float32))]}),
    ('Conv2D_1', {
        'block': P.Conv2D(1, 6, pad_mode='same', pad=0),
        'desc_inputs': [Tensor(np.ones(shape=[5, 5, 8, 8]).astype(np.float32)),
                        Tensor(np.ones(shape=[1, 5, 6, 6]).astype(np.float32))]}),
    ('Conv2D_2', {
        'block': P.Conv2D(1, 6, pad_mode='valid', pad=0),
        'desc_inputs': [Tensor(np.ones(shape=[5, 5, 8, 8]).astype(np.float32)),
                        Tensor(np.ones(shape=[1, 5, 6, 6]).astype(np.float32))]}),
    ('Conv2D_3', {
        'block': P.Conv2D(1, 6, pad_mode='pad', pad=0),
        'desc_inputs': [Tensor(np.ones(shape=[5, 5, 8, 8]).astype(np.float32)),
                        Tensor(np.ones(shape=[1, 5, 6, 6]).astype(np.float32))]}),
    ('Conv2D_4', {
        'block': P.Conv2D(1, 6, pad_mode='pad', pad=7),
        'desc_inputs': [Tensor(np.ones(shape=[5, 5, 8, 8]).astype(np.float32)),
                        Tensor(np.ones(shape=[1, 5, 6, 6]).astype(np.float32))]}),
    ('MatMul_1', {
        'block': P.MatMul(),
        'desc_inputs': [Tensor(np.ones(shape=[1, 3])), Tensor(np.ones(shape=[3, 4]))]}),
    ('MatMul_2', {
        'block': P.BatchMatMul(),
        'desc_inputs': [Tensor(np.ones(shape=[5, 1, 5])), Tensor(np.ones(shape=[5, 5, 4]))]}),
    ('MatMul_Transpose_1', {
        'block': P.MatMul(transpose_a=True),
        'desc_inputs': [Tensor(np.ones(shape=[3, 1])), Tensor(np.ones(shape=[3, 4]))]}),
    ('MatMul_Transpose_2', {
        'block': P.MatMul(transpose_b=True),
        'desc_inputs': [Tensor(np.ones(shape=[3, 2])), Tensor(np.ones(shape=[5, 2]))]}),
    ('MatMul_Transpose_3', {
        'block': P.MatMul(transpose_a=True, transpose_b=True),
        'desc_inputs': [Tensor(np.ones(shape=[3, 2])), Tensor(np.ones(shape=[5, 3]))]}),
    ('BatchMatMul', {
        'block': NetBatchMatMul(),
        'desc_inputs': [Tensor(np.ones(shape=[3, 1, 5])), Tensor(np.ones(shape=[3, 5, 4]))]}),
    ('BatchMatMul_broadcast_1', {
        'block': NetBatchMatMul(),
        'desc_inputs': [Tensor(np.ones(shape=[3, 1, 5])), Tensor(np.ones(shape=[5, 4]))]}),
    ('BatchMatMul_broadcast_2', {
        'block': NetBatchMatMul(),
        'desc_inputs': [Tensor(np.ones(shape=[3, 1, 5])), Tensor(np.ones(shape=[1, 5, 4]))]}),
    ('BatchMatMul_broadcast_3', {
        'block': NetBatchMatMul(),
        'desc_inputs': [Tensor(np.ones(shape=[2, 1, 1, 5])), Tensor(np.ones(shape=[1, 2, 5, 4]))]}),
    ('BatchMatMul_broadcast_4', {
        'block': NetBatchMatMul(),
        'desc_inputs': [Tensor(np.ones(shape=[2, 2, 1, 1, 5])), Tensor(np.ones(shape=[1, 2, 5, 4]))]}),
    ('BatchMatMul_broadcast_5', {
        'block': NetBatchMatMul(),
        'desc_inputs': [Tensor(np.ones(shape=[3, 1, 5])), Tensor(np.ones(shape=[1, 3, 5, 4]))]}),
]

test_case_lists = [test_case_check_ops]
test_exec_case = functools.reduce(lambda x, y: x + y, test_case_lists)
# use -k to select certain testcast
# pytest tests/python/ops/test_ops.py::test_backward -k LayerNorm



@non_graph_engine
@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_exec_case


raise_set = [
    ('Conv_Padding_1_Error', {
        'block': (lambda x: nn.Conv2d(1, 6, 5, pad_mode='same', padding=7), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[1, 1, 6, 5]).astype(np.float32))]}),
    ('Conv_Padding_2_Error', {
        'block': (lambda x: nn.Conv2d(1, 6, 5, pad_mode='same', padding=7), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[1, 1, 6, 5]).astype(np.float32))]}),
    ('Conv2D_1_Error', {
        'block': (lambda x, y: P.Conv2D(1, 6, pad_mode='same', pad=7), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[5, 5, 8, 8]).astype(np.float32)),
                        Tensor(np.ones(shape=[1, 5, 6, 6]).astype(np.float32))]}),
    ('Conv2D_2_Error', {
        'block': (lambda x, y: P.Conv2D(1, 6, pad_mode='valid', pad=7), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[5, 5, 8, 8]).astype(np.float32)),
                        Tensor(np.ones(shape=[1, 5, 6, 6]).astype(np.float32))]}),
    ('NetAddN_Error', {
        'block': (NetAddN(), {'exception': TypeError}),
        'desc_inputs': [(np.random.randn(1, 2, 3, 4).astype(np.float32),
                         np.random.randn(1, 2, 3, 4).astype(np.float32))]}),
    ('AddN_Error', {
        'block': (P.AddN(), {'exception': TypeError}),
        'desc_inputs': [(np.random.randn(1, 2, 3, 4).astype(np.float32),
                         np.random.randn(1, 2, 3, 4).astype(np.float32))]}),
    ('Splite_Error', {
        'block': (NetSplit(), {'exception': TypeError}),
        'desc_inputs': [None]}),
    ('MatMul_1_Error', {
        'block': (P.MatMul(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[5])), Tensor(np.ones(shape=[4]))]}),
    ('MatMul_2_Error', {
        'block': (P.MatMul(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[1, 5])), Tensor(np.ones(shape=[3, 4]))]}),
    ('MatMul_3_Error', {
        'block': (P.MatMul(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[1, 5])), Tensor(np.ones(shape=[5, 5, 4]))]}),
    ('MatMul_Transpose_1_Error', {
        'block': (P.MatMul(transpose_a=True), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[1, 3])), Tensor(np.ones(shape=[3, 4]))]}),
    ('MatMul_Transpose_2_Error', {
        'block': (P.MatMul(transpose_b=True), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[3, 2])), Tensor(np.ones(shape=[2, 5]))]}),
    ('MatMul_Transpose_3_Error', {
        'block': (P.MatMul(transpose_a=True, transpose_b=True), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[3, 2])), Tensor(np.ones(shape=[3, 5]))]}),
    ('BatchMatMul_1_Error', {
        'block': (P.BatchMatMul(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[5])), Tensor(np.ones(shape=[4]))]}),
    ('BatchMatMul_2_Error', {
        'block': (P.BatchMatMul(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[1, 5])), Tensor(np.ones(shape=[3, 4]))]}),
    ('BatchMatMul_3_Error', {
        'block': (P.BatchMatMul(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[3, 1, 5])), Tensor(np.ones(shape=[3, 3, 4]))]}),
    ('BatchMatMul_4_Error', {
        'block': (P.BatchMatMul(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones(shape=[3, 1, 5])), Tensor(np.ones(shape=[2, 5, 4]))]}),
]


@mindspore_test(pipeline_for_verify_exception_for_case_by_case_config)
def test_check_exception():
    return raise_set
