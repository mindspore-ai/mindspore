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
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as P
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.ops.functional import vmap


class OneHotNet(Cell):
    def __init__(self):
        super(OneHotNet, self).__init__()
        self.onehot = P.OneHot()

    def construct(self, indices, depth, on_value, off_value):
        res = self.onehot(indices, depth, on_value, off_value)
        return res


def one_hot_static_shape_test_case(in_type, value_type, out_type):
    depth = 5
    indices = Tensor(np.array([1, 3, 2, 4, 0]).astype(in_type))
    on_value = Tensor(1.0, value_type)
    off_value = Tensor(0.0, value_type)
    net = OneHotNet()
    output = net(indices, depth, on_value, off_value)
    expect = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0, 0.0, 0.0]]).astype(out_type)

    assert np.allclose(expect, output.asnumpy(), 1.e-4, 1.e-7)


def one_hot_functional(in_type, value_type, out_type):
    depth = 5
    indices = Tensor(np.array([1, 3, 2, 4, 0]).astype(in_type))
    on_value = Tensor(1.0, value_type)
    off_value = Tensor(0.0, value_type)
    output = ops.one_hot(indices, depth, on_value, off_value)
    expect = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0, 0.0, 0.0]]).astype(out_type)

    assert np.allclose(expect, output.asnumpy(), 1.e-4, 1.e-7)


class OneHotDynamicShapeNet(Cell):
    def __init__(self):
        super(OneHotDynamicShapeNet, self).__init__()
        self.onehot = P.OneHot()
        self.unique = P.Unique()

    def construct(self, indices, depth, on_value, off_value):
        real_input, _ = self.unique(indices)
        res = self.onehot(real_input, depth, on_value, off_value)
        return res


def one_hot_dynamic_shape_test_case(in_type, value_type, out_type):
    depth = 5
    indices = Tensor(np.array([1, 3, 1, 2, 2, 4, 0, 4]).astype(in_type))
    on_value = Tensor(1.0, value_type)
    off_value = Tensor(0.0, value_type)
    net = OneHotDynamicShapeNet()
    output = net(indices, depth, on_value, off_value)
    expect = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0, 0.0, 0.0]]).astype(out_type)

    assert np.allclose(expect, output.asnumpy(), 1.e-4, 1.e-7)


def one_hot_vmap(in_type, value_type):
    def cal_onehot(ind, dep, on_v, off_v):
        return P.OneHot()(ind, dep, on_v, off_v)
    depth = 5
    on_value = Tensor(1.0, value_type)
    off_value = Tensor(0.0, value_type)
    indices = Tensor(np.array([[1, 3, 2, 4, 0], [1, 3, 2, 4, 0]]).astype(in_type))
    outputs = vmap(cal_onehot, in_axes=(0, None, None, None), out_axes=0)(indices, depth, on_value, off_value)

    expect = np.array([[[0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0]],
                       [[0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0]]]).astype(in_type)
    assert np.allclose(expect, outputs.asnumpy(), 1.e-4, 1.e-7)


def one_hot_static_shape_all_types():
    one_hot_static_shape_test_case(np.uint8, mstype.int8, np.int8)
    one_hot_static_shape_test_case(np.uint8, mstype.uint8, np.uint8)
    one_hot_static_shape_test_case(np.uint8, mstype.int32, np.int32)
    one_hot_static_shape_test_case(np.uint8, mstype.float16, np.float16)
    one_hot_static_shape_test_case(np.uint8, mstype.float32, np.float32)
    one_hot_static_shape_test_case(np.int32, mstype.int8, np.int8)
    one_hot_static_shape_test_case(np.int32, mstype.uint8, np.uint8)
    one_hot_static_shape_test_case(np.int32, mstype.int32, np.int32)
    one_hot_static_shape_test_case(np.int32, mstype.float16, np.float16)
    one_hot_static_shape_test_case(np.int32, mstype.float32, np.float32)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ascend_graph_mode():
    """
    Feature: test one_hot static shape on Ascend in graph mode
    Description: test interface
    Expectation: result match numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    one_hot_static_shape_all_types()
    one_hot_dynamic_shape_test_case(np.int32, mstype.float32, np.float32)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ascend_pynative_mode():
    """
    Feature: test one_hot static shape on Ascend in pynative mode
    Description: test interface pynative mode
    Expectation: result match numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    one_hot_static_shape_all_types()
    one_hot_dynamic_shape_test_case(np.int32, mstype.float32, np.float32)
    one_hot_functional(np.int32, mstype.float32, np.float32)
    one_hot_vmap(np.int32, mstype.float32)
