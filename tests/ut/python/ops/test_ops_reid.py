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
""" test Activations """
import functools
import numpy as np

import mindspore.nn as nn
from mindspore.ops import operations as P
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config
from ....mindspore_test_framework.pipeline.gradient.compile_gradient \
    import pipeline_for_compile_grad_ge_graph_for_case_by_case_config
from ....ops_common import convert


class SeqConvBnRelu(nn.Cell):
    """ SeqConvBnRelu definition """

    def __init__(self, in_ch, out_ch):
        super(SeqConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = P.ReLU()

    def construct(self, input_x):
        return self.relu(self.bn(self.conv(input_x)))


test_case_reid_ops = [
    ('ReduceMax', {
        'block': P.ReduceMax(keep_dims=False),
        'desc_const': [(1,)],
        'desc_inputs': [convert([32, 32], np.float16)],
        'desc_bprop': [convert([32], np.float16)],
        'skip': []}),
    ('ReduceMin', {
        'block': P.ReduceMin(),
        'desc_const': [(1,)],
        'desc_inputs': [[32, 32]],
        'desc_bprop': [[32]],
        'skip': []}),
    ('ReduceMean', {
        'block': P.ReduceMean(keep_dims=True),
        'desc_const': [(1, 2)],
        'desc_inputs': [[32, 4, 4]],
        'desc_bprop': [[32, 1, 1]]}),
    ('Log', {
        'block': P.Log(),
        'desc_inputs': [[4, 128, 1024]],
        'desc_bprop': [[4, 128, 1024]],
        'skip': ['backward']}),  # check backward error
    ('Reciprocal', {
        'block': P.Reciprocal(),
        'desc_inputs': [[4, 128, 1024]],
        'desc_bprop': [[4, 128, 1024]],
        'skip': ['backward']}),
    ('FloorDiv', {
        'block': P.FloorDiv(),
        'desc_inputs': [[4, 128, 1024], [4, 128, 1024]],
        'desc_bprop': [[4, 128, 1024]]}),
    ('Sigmoid', {
        'block': P.Sigmoid(),
        'desc_inputs': [[4, 128, 1024]],
        'desc_bprop': [[4, 128, 1024]]}),
    ('Softmax', {
        'block': P.Softmax(),
        'desc_inputs': [[1, 16]],
        'desc_bprop': [[1, 16]],
        'skip': ['backward']}),  # check backward error
    ('Softmax', {
        'block': P.Softmax(axis=(0, 1)),
        'desc_inputs': [[1, 16]],
        'desc_bprop': [[1, 16]],
        'skip': ['backward']}),
    ('L2Normalize', {
        'block': P.L2Normalize(),
        'desc_inputs': [[4, 128, 1024]],
        'desc_bprop': [[4, 128, 1024]]}),
    ('ReLU', {
        'block': P.ReLU(),
        'desc_inputs': [[64, 64, 112, 112]],
        'desc_bprop': [[64, 64, 112, 112]]}),
    ('SeqConvBnRelu', {
        'block': SeqConvBnRelu(3, 64),
        'desc_inputs': [[64, 3, 112, 112]],
        'desc_bprop': [[64, 64, 112, 112]]}),
    ('PReluCell', {
        'block': nn.PReLU(1, [np.float32(0.25)]),
        'desc_inputs': [[128, 64, 112, 112]],
        'desc_bprop': [[128, 64, 112, 112]]}),
    ('PRelu', {
        'block': P.PReLU(),
        'desc_inputs': [[128, 64, 112, 112], [64,]],
        'desc_bprop': [[128, 64, 112, 112]]}),
    ('Cos', {
        'block': P.Cos(),
        'desc_inputs': [[8, 16]],
        'desc_bprop': [[8, 16]]}),
    ('ACos', {
        'block': P.ACos(),
        'desc_inputs': [[8, 16]],
        'desc_bprop': [[8, 16]]}),
    ('Exp', {
        'block': P.Exp(),
        'desc_inputs': [[256, 8]],
        'desc_bprop': [[256, 8]]}),
    ('Pow', {
        'block': P.Pow(),
        'desc_const': [2.0],
        'desc_inputs': [[1, 512]],
        'desc_bprop': [[1, 512]]}),
    ('LogicalNot', {
        'block': P.LogicalNot(),
        'desc_inputs': [convert([256], np.bool_)],
        'desc_bprop': [convert([256], np.bool_)]}),
    ('Equal', {
        'block': P.Equal(),
        'desc_inputs': [convert([256], np.float16), convert([256], np.float16)],
        'desc_bprop': [convert([256], np.bool_)]}),
    ('Greater', {
        'block': P.Greater(),
        'desc_inputs': [convert([256], np.float16), convert([256], np.float16)],
        'desc_bprop': [convert([256], np.bool_)]}),
    ('Dropout', {
        'block': nn.Dropout(p=0.5),
        'desc_inputs': [[1, 512, 7, 7]],
        'desc_bprop': [[1, 512, 7, 7]]}),
    ('MatMul', {
        'block': P.MatMul(),
        'desc_inputs': [[64, 512], [512, 64]],
        'desc_bprop': [[64, 64]]}),
    ('Maximum', {
        'block': P.Maximum(),
        'desc_inputs': [[64, 1], [64, 1]],
        'desc_bprop': [[64, 1]]}),
]

test_case_lists = [test_case_reid_ops]
test_case = functools.reduce(lambda x, y: x + y, test_case_lists)
# use -k to select certain testcast
# pytest  tests/python/ops/test_ops.py::test_backward -k LayerNorm


test_exec_case = filter(lambda x: 'skip' not in x[1] or
                        'exec' not in x[1]['skip'], test_case)

test_backward_exec_case = filter(lambda x: 'skip' not in x[1] or
                                 'backward' not in x[1]['skip'] and 'backward_exec'
                                 not in x[1]['skip'], test_case)


@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    return test_exec_case


@mindspore_test(pipeline_for_compile_grad_ge_graph_for_case_by_case_config)
def test_backward_exec():
    return test_backward_exec_case
