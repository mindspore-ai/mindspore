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
""" test graph joinedstr """
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, jit, context
import mindspore.ops as ops
import mindspore.ops.operations as P

context.set_context(mode=context.GRAPH_MODE)


def test_joinedstr_basic_tuple_list_dict():
    """
    Feature: Support joinedstr.
    Description: Support joinedstr.
    Expectation: No exception.
    """
    @jit
    def joined_net():
        x = (1, 2, 3, 4, 5)
        y = [1, 2, 3, 4, 5]
        z = {'a': 1, 'b': 2, 'c': 3}
        res_x = f"x: {x}"
        res_y = f"y: {y}"
        res_z = f"z: {z}"
        return res_x, res_y, res_z

    out_x, out_y, out_z = joined_net()
    assert out_x == "x: (1, 2, 3, 4, 5)"
    assert out_y == "y: [1, 2, 3, 4, 5]"
    assert out_z == "z: {'a': 1, 'b': 2, 'c': 3}"


def test_joinedstr_basic_dict_key():
    """
    Feature: Support joinedstr.
    Description: Support joinedstr.
    Expectation: No exception.
    """
    @jit
    def joined_net():
        c = (1, 2, 3, 4, 5)
        dict_key = f"c: {c}"
        z = {'a': 1, 'b': 2, dict_key: 3}
        dict_res = z.get(dict_key)
        return dict_res

    out = joined_net()
    assert out == 3


def test_joinedstr_basic_numpy_scalar():
    """
    Feature: Support joinedstr.
    Description: Support joinedstr.
    Expectation: No exception.
    """
    @jit
    def joined_net():
        x = np.array([1, 2, 3, 4, 5])
        y = 3
        res = f"x: {x}, y: {y}"
        return res

    out = joined_net()
    assert out == "x: [1 2 3 4 5], y: 3"


def test_joinedstr_inner_tensor():
    """
    Feature: Support joinedstr.
    Description: Support joinedstr.
    Expectation: No exception.
    """
    @jit
    def joined_net():
        x = (1, 2, 3, 4, 5)
        inner_tensor_1 = Tensor(x)
        res = f"x: {x}, inner_tensor_1: {inner_tensor_1}, inner_tensor_2: {Tensor(2)}"
        return res

    out = joined_net()
    assert str(
        out) == "x: (1, 2, 3, 4, 5), inner_tensor_1: [1 2 3 4 5], inner_tensor_2: 2"


def test_joinedstr_dynamic_shape_scalar():
    """
    Feature: Support joinedstr.
    Description: dynamic shape is scalar in joined str.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.unique = P.Unique()
            self.gather = P.Gather()
            self.axis = 0
            self.shape = ops.TensorShape()

        def construct(self, x, indices):
            unique_indices, _ = self.unique(indices)
            x = self.gather(x, unique_indices, self.axis)
            x_shape = self.shape(x)
            print(f"x.shape:{x_shape}")
            return x_shape

    input_np = Tensor(np.random.randn(5, 4).astype(np.float32))
    indices_np = Tensor(np.random.randint(0, 3, size=3))
    net = Net()
    out = net(input_np, indices_np)
    print("out:", out)
