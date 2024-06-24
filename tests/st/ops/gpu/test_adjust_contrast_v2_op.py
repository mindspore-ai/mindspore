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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.context as context
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore.ops.operations.image_ops import AdjustContrastv2
from mindspore.common import dtype as mstype


class AdjustContrastV2(Cell):
    def __init__(self):
        super().__init__()
        self.adjustcontrastv2 = AdjustContrastv2()

    def construct(self, images, contrast_factor):
        return self.adjustcontrastv2(images, contrast_factor)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_adjust_contrast_v2():
    """
    Feature:  AdjustContrastV2 2 inputs and 1 output.
    Description: compute result of AdjustContrastV2.
    Expectation: The result matches numpy implementation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    images = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
    contrast_factor = np.array(2)
    expected_out = np.array([[[-3.5, -2.5, -1.5], [2.5, 3.5, 4.5]], [[8.5, 9.5, 10.5], [14.5, 15.5, 16.5]]])

    net = AdjustContrastV2()
    out = net(Tensor(images, dtype=mstype.float32), Tensor(contrast_factor, dtype=mstype.float32))
    np.testing.assert_almost_equal(out.asnumpy(), expected_out)
