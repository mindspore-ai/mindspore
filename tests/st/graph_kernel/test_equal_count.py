# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.ops import operations as P


class EqualCount(nn.Cell):
    def __init__(self):
        super(EqualCount, self).__init__()
        self.op = P.EqualCount()

    def construct(self, *inp):
        return self.op(*inp)


def get_output(*inp, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    output = EqualCount()(*inp)
    return output


def basic_test(datatype):
    x = Tensor(np.array([[1, 1, 1, 1], [3, 3, 3, 3]]).astype(datatype))
    y = Tensor(np.array([[1, 2, 1, 2], [1, 1, 3, 3]]).astype(datatype))
    expect = get_output(x, y, enable_graph_kernel=False)
    output = get_output(x, y, enable_graph_kernel=True)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1.e-4, 1.e-7)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gpu_fp16():
    """
    Feature: equalcount op open graphkernel on gpu
    Description: equalcount op on gpu set graph mode test open graph kernel flag
    Expectation: open graph kernel result equal to close graph kernel
    """
    context.set_context(mode=context.GRAPH_MODE)
    basic_test(np.float16)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gpu_fp32():
    """
    Feature: equalcount op open graphkernel on gpu
    Description: equalcount op on gpu set graph mode test open graph kernel flag
    Expectation: open graph kernel result equal to close graph kernel
    """
    context.set_context(mode=context.GRAPH_MODE)
    basic_test(np.float32)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ascend_pynative_mode_fp32():
    """
    Feature: equalcount op expand fallback on ascend
    Description: equalcount op on ascend set pynative mode test expand fallback
    Expectation: open graph kernel result equal to close graph kernel
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    basic_test(np.float32)
