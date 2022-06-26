# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import operations as P


class NetDynInput(nn.Cell):
    def __init__(self, shape):
        super(NetDynInput, self).__init__()
        self.scatternd = P.ScatterNd()
        self.shape = shape

    def construct(self, indices, update):
        return self.scatternd(indices, update, self.shape)


class NetDynShape(nn.Cell):
    def __init__(self):
        super(NetDynShape, self).__init__()
        self.scatternd = P.ScatterNd()
        self.shape_op = P.TensorShape()

    def construct(self, indices, update, prev_out):
        shape = self.shape_op(prev_out)
        return self.scatternd(indices, update, shape)


def check_result(output, expect):
    error = np.ones(shape=output.shape) * 1.0e-6
    diff = output - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


def case_dyn_input():
    indices = np.array(
        [[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]).astype(np.int32)
    update = np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(np.float32)
    shape = (2, 2)
    expect = np.array([[0., 5.3],
                       [0., 1.1]]).astype(np.float32)
    net = NetDynInput(shape)
    indices_dyn = Tensor(shape=[None, 2], dtype=mstype.int32)
    update_dyn = Tensor(shape=[None], dtype=mstype.float32)
    net.set_inputs(indices_dyn, update_dyn)
    output = net(Tensor(indices), Tensor(update)).asnumpy()
    check_result(output, expect)


def case_dyn_shape():
    indices = np.array(
        [[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]).astype(np.int32)
    update = np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(np.float32)
    prev_out = np.array([[1, 1],
                         [1, 1]]).astype(np.int32)
    expect = np.array([[0., 5.3],
                       [0., 1.1]]).astype(np.float32)
    net = NetDynShape()
    prev_out_dyn = Tensor(shape=[None, 2], dtype=mstype.int32)
    net.set_inputs(Tensor(indices), Tensor(update), prev_out_dyn)
    output = net(Tensor(indices), Tensor(update), Tensor(prev_out)).asnumpy()
    check_result(output, expect)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_scatternd_dyn_input():
    """
    Feature: dynamic shape for ScatterNd
    Description: dynamic input shape for ScatterNd
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE)
    case_dyn_input()
    context.set_context(mode=context.PYNATIVE_MODE)
    case_dyn_input()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_scatternd_dyn_shape():
    """
    Feature: dynamic shape for ScatterNd
    Description: dynamic output shape for ScatterNd when shape is a tensor
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE)
    case_dyn_shape()
    context.set_context(mode=context.PYNATIVE_MODE)
    case_dyn_shape()
