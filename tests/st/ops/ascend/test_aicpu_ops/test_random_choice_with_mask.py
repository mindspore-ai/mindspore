# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P


class RandomChoiceWithMaskNet(nn.Cell):
    def __init__(self, count):
        super().__init__()
        self.random_choice_with_mask = P.RandomChoiceWithMask(
            count=count, seed=1)

    def construct(self, x):
        return self.random_choice_with_mask(x)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_random_choice_with_mask_check_value(mode):
    """
    Feature: Custom aicpu feature.
    Description: Test random_choice_with_mask kernel.
    Expectation: Output value is the same as expected.
    """
    context.set_context(mode=mode, device_target="Ascend")
    # Sample all
    x = np.array([[1, 1],
                  [1, 1]]).astype(np.bool)
    expect_coordinate = set([(0, 0), (0, 1), (1, 0), (1, 1)])
    expect_mask = np.array([True, True, True, True], np.bool)
    net = RandomChoiceWithMaskNet(4)
    coordinate, mask = net(Tensor(x))
    coordinate_set = set(tuple(coord) for coord in coordinate.asnumpy())
    assert coordinate_set == expect_coordinate
    assert np.all(mask.asnumpy() == expect_mask)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_random_choice_with_mask(mode):
    """
    Feature: Custom aicpu feature.
    Description: Test random_choice_with_mask kernel.
    Expectation: Output shape is the same as expected.
    """
    context.set_context(mode=mode, device_target="Ascend")
    x = np.array([[1, 0, 1],
                  [1, 1, 0]]).astype(np.bool)
    count = 3
    net = RandomChoiceWithMaskNet(count)
    coordinate, mask = net(Tensor(x))
    assert coordinate.shape == (count, len(x.shape))
    assert mask.shape == (count,)
