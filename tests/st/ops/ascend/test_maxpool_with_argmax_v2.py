# Copyright 2023 Huawei Technologies Co., Ltd
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

import mindspore.ops.operations as P
from mindspore import context, Tensor
from mindspore.nn import Cell
from mindspore import dtype as mstype

context.set_context(device_target="Ascend")


class MaxPoolWithArgmaxV2Net(Cell):
    def __init__(self, kernel_size, strides, pads, dilation, ceil_mode, argmax_type=mstype.int64):
        super(MaxPoolWithArgmaxV2Net, self).__init__()
        self.maxpool_with_argmax_v2 = P.MaxPoolWithArgmaxV2(kernel_size, strides, pads, dilation, ceil_mode,
                                                            argmax_type)

    def construct(self, input_data):
        output, argmax = self.maxpool_with_argmax_v2(input_data)
        return output, argmax


def test_maxpool_with_argmax_v2_ceil_mode_false():
    """
    Feature: Test MaxPoolWithArgmaxV2.
    Description: Test MaxPoolWithArgmaxV2 with float32 inputs.
    Expectation: success.
    """
    attributes = {'kernel_size': (3, 2), 'strides': (2, 1), 'pads': 0, 'dilation': 1,
                  'ceil_mode': False, 'argmax_type': mstype.int32}
    x = Tensor(np.arange(20 * 16 * 50 * 32).reshape((20, 16, 50, 32)), mstype.float32)
    net = MaxPoolWithArgmaxV2Net(**attributes)
    output, _ = net(x)
    print("***********output output*********")
    print(output.asnumpy())


def test_maxpool_with_argmax_v2_ceil_mode_true():
    """
    Feature: Test MaxPoolWithArgmaxV2.
    Description: Test MaxPoolWithArgmaxV2 with `ceil_mode` is True.
    Expectation: success.
    """
    attributes = {'kernel_size': (3, 2), 'strides': (2, 1), 'pads': 0, 'dilation': 1,
                  'ceil_mode': True, 'argmax_type': mstype.int32}
    x = Tensor(np.arange(20 * 16 * 50 * 32).reshape((20, 16, 50, 32)), mstype.float32)
    net = MaxPoolWithArgmaxV2Net(**attributes)
    output, _ = net(x)
    print("***********output output*********")
    print(output.asnumpy())
