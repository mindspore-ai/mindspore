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
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops.operations._grad_ops import ResizeBicubicGrad
import pytest


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_resize_bicubic_grad_graph():
    """
    Feature: test operations in result and output type
    Description: test in graph mode on GPU
    Expectation: success or throw pytest error
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    types = [np.float32, np.float64]
    for type_i in types:
        grad = np.array([1, 2, 3, 4])
        grad = grad.reshape([1, 1, 2, 2])
        gradients = Tensor(grad.astype(type_i))
        ori = np.array([1, 2, 3, 4])
        ori = ori.reshape([1, 1, 1, 4])
        origin = Tensor(ori.astype(type_i))
        output = ResizeBicubicGrad(False, False)(gradients, origin)
        expect_type = output.asnumpy().dtype
        expect = np.array([4, 0, 6, 0])
        expect = expect.reshape([1, 1, 1, 4])

        expect = expect.astype(type_i)
        assert expect_type == type_i
        assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_resize_bicubic_grad_pynative():
    """
    Feature: test operations in result and output type
    Description: test in pynative mode on GPU
    Expectation: success or throw pytest error
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    types_2 = [np.float32, np.float64]
    for type_i in types_2:
        grad = np.array([1, 2, 3, 4])
        grad = grad.reshape([1, 1, 2, 2,])
        gradients = Tensor(grad.astype(type_i))
        ori = np.array([1, 2, 3, 4])
        ori = ori.reshape([1, 1, 1, 4])
        origin = Tensor(ori.astype(type_i))
        output = ResizeBicubicGrad(False, False)(gradients, origin)
        expect_type_2 = output.asnumpy().dtype
        expect = np.array([4, 0, 6, 0])
        expect = expect.reshape([1, 1, 1, 4])

        expect = expect.astype(type_i)
        assert expect_type_2 == type_i
        assert (output.asnumpy() == expect).all()
