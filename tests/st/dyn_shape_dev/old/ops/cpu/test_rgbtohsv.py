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
import mindspore as ms
from mindspore import nn, Tensor, context
from mindspore.ops.operations.image_ops import RGBToHSV


class Net(nn.Cell):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.op = RGBToHSV()

    def construct(self, images):
        return self.op(images)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_rgb_to_csv_dyn():
    """
    Feature: test RGBToHSV ops in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    net = Net()
    images_dyn = Tensor(shape=[None, None, None, 3], dtype=ms.float32)
    net.set_inputs(images_dyn)
    images = Tensor(
        np.array([0.25, 0.5, 0.5]).astype(np.float32).reshape([1, 1, 1, 3]))
    output = net(images)

    expect_shape = (1, 1, 1, 3)
    assert output.asnumpy().shape == expect_shape
