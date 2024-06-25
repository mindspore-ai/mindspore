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
import pytest
import numpy as np
from mindspore import Tensor, jit, context, nn
import mindspore.ops as ops
import mindspore.ops.operations as P
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_joinedstr_basic_variable_gpu():
    """
    Feature: Support joinedstr.
    Description: Support joinedstr.
    Expectation: No exception.
    """
    @jit
    def joined_net(x, y):
        if (x > 2 * y).all():
            res = f"res: {2 * y}"
        else:
            res = f"res: {x}"
        return res

    input_x = Tensor(np.array([1, 2, 3, 4, 5]))
    out = joined_net(input_x, input_x)
    result_tensor = Tensor(np.array([1, 2, 3, 4, 5]))
    assert out == f"res: {result_tensor}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_joinedstr_basic_variable_ascend():
    """
    Feature: Support joinedstr.
    Description: Support joinedstr.
    Expectation: No exception.
    """
    @jit
    def joined_net(x, y):
        if (x > 2 * y).all():
            res = f"res: {2 * y}"
        else:
            res = f"res: {x}"
        return res


    input_x = Tensor(np.array([1, 2, 3, 4, 5]))
    out = joined_net(input_x, input_x)
    assert out == "res: [1 2 3 4 5]"



@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_joinedstr_basic_variable_2():
    """
    Feature: Support joinedstr.
    Description: Support joinedstr.
    Expectation: No exception.
    """
    @jit
    def joined_net(x, y):
        if (x > 2 * y).all():
            res = f"{2 * y}"
        else:
            res = f"{x}"
        return res

    input_x = Tensor(np.array([1, 2, 3, 4, 5]))
    out = joined_net(input_x, input_x)
    assert str(out) == "[1 2 3 4 5]"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_joinedstr_out_tensor():
    """
    Feature: Support joinedstr.
    Description: Support joinedstr.
    Expectation: No exception.
    """
    @jit
    def joined_net(x):
        return f"x: {x}"

    input_x = Tensor([1, 2, 3])
    out = joined_net(input_x)
    assert str(out) == "x: [1 2 3]"


@pytest.mark.skip("the print output when I6WM9U has been resolved."
                  "otherwise the print may cause sync stream error because the data has not been sync to device")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
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
            self.shape = ops.Shape()

        def construct(self, x, indices):
            unique_indices, _ = self.unique(indices)
            x = self.gather(x, unique_indices, self.axis)
            x_shape = self.shape(x)
            # TODO(@lianliguang): Check the print output when I6WM9U has been resolved
            print(f"x.shape:{x_shape}")
            return x_shape

    input_np = Tensor(np.random.randn(5, 4).astype(np.float32))
    indices_np = Tensor(np.random.randint(0, 3, size=3))
    net = Net()
    out = net(input_np, indices_np)
    print("out:", out)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_joinedstr_with_nested_pyinterpret():
    """
    Feature: Support joinedstr.
    Description: Support joinedstr.
    Expectation: No exception.
    """
    @jit
    def joined_net(x):
        return f"{type(x).__name__}"

    input_x = Tensor([1, 2, 3])
    out = joined_net(input_x)
    assert str(out) == "Tensor"
