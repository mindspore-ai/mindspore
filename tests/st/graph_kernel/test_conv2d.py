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
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore.common.initializer import initializer
from mindspore import Tensor


class Conv2dNet2(nn.Cell):
    """ a seq of  conv2d Net with pad mode 'same' """

    def __init__(self, auto_prefix=True, flags=None):
        super(Conv2dNet2, self).__init__()
        self.weight = initializer(
            'normal', shape=(32, 32, 3, 3))
        self.conv2d = nn.Conv2d(
            32, 32, 3, 2, pad_mode='same', weight_init=self.weight)
        self.conv2d2 = nn.Conv2d(
            32, 32, 1, 1)
        self.conv2d3 = nn.Conv2d(
            32, 32, 1, 1)

    def construct(self, x):
        temp1 = self.conv2d(x)
        temp2 = self.conv2d2(temp1)
        temp3 = self.conv2d3(temp2)
        return temp3


def get_output(net, inp, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    output = net(inp)
    return output


def basic_test(net, inp):
    expect = get_output(net, inp, True)
    output = get_output(net, inp, False)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1.e-4, 1.e-7)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conv_fp16():
    """
    Feature: Graph Kernel expander
    Description: Verify Conv2D expander in GPU
    Expectation: No exception
    """
    context.set_context(mode=context.GRAPH_MODE, graph_kernel_flags='--enable_expand_ops=Conv2D')
    inp = Tensor(np.random.random((32, 32, 52, 52)).astype(np.float32))
    basic_test(Conv2dNet2().to_float(ms.float16), inp)
