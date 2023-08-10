# Copyright 2021 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore.common.initializer import initializer
from mindspore import Tensor


class Conv2d1Net(nn.Cell):
    """ a simple conv2d Net"""

    def __init__(self):
        super(Conv2d1Net, self).__init__()
        self.weight = initializer(
            'ones', shape=(32, 1, 1, 32), dtype=ms.float16)
        self.conv2d = nn.Conv2d(
            32, 32, 1, pad_mode='valid', weight_init=self.weight, data_format="NHWC")

    def construct(self, x):
        return self.conv2d(x)


def get_output(net, inp, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    output = net()(inp)
    return output


def basic_test(net, datatype):
    inp = Tensor(np.random.random((32, 16, 16, 32)).astype(datatype))
    expect = get_output(net, inp, False)
    output = get_output(net, inp, True)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1.e-4, 1.e-7)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_fp16():
    """
    Feature: Graph Kernel expander
    Description: Verify Conv2D expander in GPU
    Expectation: No exception
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU",
                        graph_kernel_flags='--enable_expand_ops=Conv2D')
    basic_test(Conv2d1Net, np.float16)
