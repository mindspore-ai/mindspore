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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, input_size, spatial_scale, group_size, output_dim):
        super(Net, self).__init__()
        self.roi_pooling = G.PSROIPoolingGrad(input_size, spatial_scale, group_size, output_dim)

    @jit
    def construct(self, x, rois):
        return self.roi_pooling(x, rois)


def test_net(x_shape, rois_shape, input_size, spatial_scale, group_size, output_dim):
    """
    Feature: test PSROIPoolingGrad.
    Description:
        Input:
            x: shape is: [n, c, out_shape, out_shape].
            rois: shape is: [n1, c2, n2], where c2 is 5, n1 represent the batch size,
                  and n2 = n // n1 (also mean n = n1 * n2).

            input_size: should contain 2 value, refers to width and height, refers to h1 and w1 in output shape.
            spatial_scale: default is 1/16.0.
            group_size: should equal to out_shape.
            output_dim: (output_dim + C0 - 1) // C0 == c, where c0 is 16 in davinci.

        output:
            output shape: [n1, c1, h1, w1], where n1 is the batch_size in rois, c1 = c * out_shape * out_shape
                      h1 and w1 is output resolution.
    Expectation: Run successfully.
    """

    np_x = np.random.random(x_shape).astype(np.float32)
    input_x = Tensor(np_x)

    np_rois = np.random.random(rois_shape).astype(np.float32)
    input_rois = Tensor(np_rois)

    roi_grad = Net(input_size, spatial_scale, group_size, output_dim)
    output = roi_grad(input_x, input_rois)
    print(output.asnumpy())


if __name__ == "__main__":
    test_net((512, 22, 7, 7), (4, 5, 128), (84, 84), 0.0625, 7, 22)
    test_net((16, 2, 7, 7), (1, 5, 16), (28, 28), 0.0625, 7, 2)
