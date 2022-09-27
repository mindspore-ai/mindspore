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

import pytest
import numpy as np
from mindspore import context, nn, Tensor
from mindspore import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)

single_element_fg = C.MultitypeFuncGraph("single_element_fg")
@single_element_fg.register("Tensor")
def single_element_fg_for_tensor(x):
    return P.Square()(x)

double_elements_fg = C.MultitypeFuncGraph("double_elements_fg")
@double_elements_fg.register("Tensor", "Tuple")
def double_elements_fg_for_tensor_tuple(x, y):
    return P.Tile()(x, y)

@double_elements_fg.register("Tensor", "List")
def double_elements_fg_for_tensor_list(x, y):
    return x + y[0]


class HyperMapNet(nn.Cell):
    def __init__(self, fg):
        super(HyperMapNet, self).__init__()
        self.common_map = C.HyperMap()
        self.fg = fg

    def construct(self, nest_tensor_list):
        output = self.common_map(self.fg, *nest_tensor_list)
        return output


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_single_element_hypermap_with_tensor_input():
    """
    Feature: HyperMap
    Description: Test whether the HyperMap with single tensor input can run successfully.
    Expectation: success.
    """
    x = (Tensor(np.array([1, 2, 3]), mstype.float32), Tensor(np.array([4, 5, 6]), mstype.float32))
    common_map = HyperMapNet(single_element_fg)
    output = common_map((x,))
    expect_output_1 = np.array([1.0, 4.0, 9.0])
    expect_output_2 = np.array([16.0, 25.0, 36.0])
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], Tensor)
    assert isinstance(output[1], Tensor)
    assert np.allclose(output[0].asnumpy(), expect_output_1)
    assert np.allclose(output[1].asnumpy(), expect_output_2)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_double_elements_hypermap_tensor_tuple_inputs():
    """
    Feature: HyperMap
    Description: Test whether the HyperMap with tensor and tuple inputs can run successfully.
    Expectation: success.
    """
    x = (Tensor(np.array([1, 2, 3]), mstype.float32), Tensor(np.array([4, 5, 6]), mstype.float32))
    y = ((1, 2), (2, 1))
    common_map = HyperMapNet(double_elements_fg)
    output = common_map((x, y))
    expect_output_1 = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    expect_output_2 = np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], Tensor)
    assert isinstance(output[1], Tensor)
    assert np.allclose(output[0].asnumpy(), expect_output_1)
    assert np.allclose(output[1].asnumpy(), expect_output_2)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_double_elements_hypermap_tensor_list_inputs():
    """
    Feature: HyperMap
    Description: Test whether the HyperMap with tensor and list inputs can run successfully.
    Expectation: success.
    """
    x = (Tensor(np.array([1, 2, 3]), mstype.float32), Tensor(np.array([4, 5, 6]), mstype.float32))
    y = ([1, 2], [2, 1])
    common_map = HyperMapNet(double_elements_fg)
    output = common_map((x, y))
    expect_output_1 = np.array([2.0, 3.0, 4.0])
    expect_output_2 = np.array([6.0, 7.0, 8.0])
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], Tensor)
    assert isinstance(output[1], Tensor)
    assert np.allclose(output[0].asnumpy(), expect_output_1)
    assert np.allclose(output[1].asnumpy(), expect_output_2)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_doubel_elements_hypermap_correct_mix_inputs():
    """
    Feature: HyperMap
    Description: Test whether the HyperMap with mix correct inputs (Tensor + Tuple and Tensor + List)
                 can run successfully.
    Expectation: success.
    """
    x = (Tensor(np.array([1, 2, 3]), mstype.float32), Tensor(np.array([4, 5, 6]), mstype.float32))
    y = ((1, 2), [2, 1])
    common_map = HyperMapNet(double_elements_fg)
    output = common_map((x, y))
    expect_output_1 = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    expect_output_2 = np.array([6.0, 7.0, 8.0])
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], Tensor)
    assert isinstance(output[1], Tensor)
    assert np.allclose(output[0].asnumpy(), expect_output_1)
    assert np.allclose(output[1].asnumpy(), expect_output_2)



@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_double_elements_hypermap_inputs_length_mismatch():
    """
    Feature: HyperMap
    Description: When the inputs to hypermap have different length, error will be raised.
    Expectation: error.
    """
    x = (Tensor(np.array([1, 2, 3]), mstype.float32), Tensor(np.array([4, 5, 6]), mstype.float32))
    y = ((1, 2), (2, 1), (5, 6))
    common_map = HyperMapNet(double_elements_fg)
    with pytest.raises(Exception, match="The length of tuples in HyperMap must be the same"):
        common_map((x, y))


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_double_elements_hypermap_inconsistent_inputs():
    """
    Feature: HyperMap
    Description: When the inputs to hypermap is inconsistent, error will be raised.
    Expectation: error.
    """
    x = (Tensor(np.array([1, 2, 3]), mstype.float32), Tensor(np.array([4, 5, 6]), mstype.float32))
    y = [(1, 2), (2, 1)]
    common_map = HyperMapNet(double_elements_fg)
    with pytest.raises(Exception, match="the types of arguments in HyperMap must be consistent"):
        common_map((x, y))
