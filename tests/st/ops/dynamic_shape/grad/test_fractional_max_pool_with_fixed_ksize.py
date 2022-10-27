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
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops.operations import nn_ops
from .test_grad_of_dynamic import TestDynamicGrad


class Net(nn.Cell):
    def __init__(self, ksize, output_shape, data_format):
        super(Net, self).__init__()
        self.func = nn_ops.FractionalMaxPoolWithFixedKsize(
            ksize, output_shape, data_format)

    def construct(self, input_x, random_samples):
        return self.func(input_x, random_samples)


def grad_dyn_case(is_dynamic_rank):
    ksize = 2
    output_shape = (2, 2)
    data_format = "NCHW"
    input_x = Tensor(np.array([3, 9, 7, 0, 3,
                               5, 5, 3, 1, 8,
                               9, 4, 9, 8, 3,
                               4, 9, 9, 6, 9,
                               9, 1, 0, 1, 9]).reshape([1, 1, 5, 5]), ms.int32)
    random_samples = Tensor(np.array([[[0.8, 0.8]]]), ms.float32)
    test_dynamic = TestDynamicGrad(Net(ksize, output_shape, data_format))
    test_dynamic.test_dynamic_grad_net(
        (input_x, random_samples), is_dynamic_rank)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_shape_fractional_max_pool_with_fixed_ksize():
    """
    Feature: FractionalMaxPoolWithFixedKsize Grad DynamicShape.
    Description: Test case of dynamic shape for  FractionalMaxPoolWithFixedKsize grad operator.
    Expectation: success.
    """
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    grad_dyn_case(False)
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(False)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_rank_fractional_max_pool_with_fixed_ksize():
    """
    Feature: FractionalMaxPoolWithFixedKsize Grad DynamicRank.
    Description: Test case of dynamic rank for  FractionalMaxPoolWithFixedKsize grad operator.
    Expectation: success.
    """
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    grad_dyn_case(True)
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(True)
