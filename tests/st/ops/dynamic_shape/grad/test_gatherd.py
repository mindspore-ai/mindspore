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
import mindspore
import mindspore.ops.operations as P
from mindspore import nn, context, Tensor
from .test_grad_of_dynamic import TestDynamicGrad

context.set_context(mode=context.PYNATIVE_MODE)


class NetGatherD(nn.Cell):
    def __init__(self):
        super(NetGatherD, self).__init__()
        self.op = P.GatherD()

    def construct(self, x, dim, index):
        return self.op(x, dim, index)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_gatherd_dynamic_shape():
    """
    Feature: GatherD Grad DynamicShape.
    Description: Test case of dynamic shape for GatherD grad operator.
    Expectation: success.
    """
    test_dynamic = TestDynamicGrad(NetGatherD(), skip_convert_out_ids=[0])
    x = Tensor(np.array([[772, 231, 508, 545, 615, 249],
                         [923, 210, 480, 696, 482, 761],
                         [465, 904, 521, 824, 607, 669],
                         [156, 539, 56, 159, 916, 566],
                         [122, 676, 714, 261, 19, 936]]), mindspore.int32)
    dim = 0
    index = Tensor(np.array([[0, 1, 0, 1, 0, -4],
                             [0, 2, 0, 2, 0, -3],
                             [0, 0, 0, 3, 3, -2],
                             [4, 4, 4, 0, 0, -1],
                             [4, 3, 2, 1, -1, -2]]), mindspore.int32)
    inputs = [x, dim, index]
    test_dynamic.test_dynamic_grad_net(inputs, False)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_gatherd_dynamic_rank():
    """
    Feature: GatherD Grad DynamicShape.
    Description: Test case of dynamic rank for GatherD grad operator on GPU.
    Expectation: success.
    """
    test_dynamic = TestDynamicGrad(NetGatherD(), skip_convert_out_ids=[0])
    x = Tensor(np.array([[772, 231, 508, 545, 615, 249],
                         [923, 210, 480, 696, 482, 761],
                         [465, 904, 521, 824, 607, 669],
                         [156, 539, 56, 159, 916, 566],
                         [122, 676, 714, 261, 19, 936]]), mindspore.int64)
    dim = 1
    index = Tensor(np.array([[0, 1, 0, 1, 0, -4],
                             [0, 2, 0, 2, 0, -3],
                             [0, 0, 0, 3, 3, -2],
                             [4, 4, 4, 0, 0, -1],
                             [4, 3, 2, 1, -1, -2]]), mindspore.int32)
    inputs = [x, dim, index]
    test_dynamic.test_dynamic_grad_net(inputs, True)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_gatherd_int_dim_dynamic_rank():
    """
    Feature: GatherD Grad DynamicShape.
    Description: Test case of dynamic rank for GatherD grad operator on GPU.
    Expectation: success.
    """
    test_dynamic = TestDynamicGrad(NetGatherD())
    x = Tensor(np.array([[772, 231, 508, 545, 615, 249],
                         [923, 210, 480, 696, 482, 761],
                         [465, 904, 521, 824, 607, 669],
                         [156, 539, 56, 159, 916, 566],
                         [122, 676, 714, 261, 19, 936]]), mindspore.int64)
    dim = 1
    index = Tensor(np.array([[0, 1, 0, 1, 0, -4],
                             [0, 2, 0, 2, 0, -3],
                             [0, 0, 0, 3, 3, -2],
                             [4, 4, 4, 0, 0, -1],
                             [4, 3, 2, 1, -1, -2]]), mindspore.int32)
    inputs = [x, dim, index]
    test_dynamic.test_dynamic_grad_net(inputs, True)
