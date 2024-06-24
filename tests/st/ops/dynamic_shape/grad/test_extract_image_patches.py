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
from mindspore import nn, context, Tensor
from mindspore.ops.operations import _inner_ops as inner
from .test_grad_of_dynamic import TestDynamicGrad


class Net(nn.Cell):
    def __init__(self, ksizes, strides, rates, padding="valid"):
        super(Net, self).__init__()
        self.extractimagepatches = inner.ExtractImagePatches(ksizes, strides, rates, padding)

    def construct(self, input_tensor):
        return self.extractimagepatches(input_tensor)


def extract_image_patches_test(is_dyn_rank):
    net = Net([1, 1, 2, 4], [1, 1, 7, 5], [1, 1, 2, 1], "valid")
    input_tensor = Tensor(np.arange(360).reshape(3, 2, 6, 10).astype(np.float32))
    tester = TestDynamicGrad(net)
    tester.test_dynamic_grad_net([input_tensor], is_dyn_rank)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_extract_image_patches_dyn_shape():
    """
    Feature: ExtractImagePatches Grad DynamicShape.
    Description: Test case of dynamic shape for ExtractImagePatches grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    extract_image_patches_test(False)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_extract_image_patches_dyn_rank():
    """
    Feature: ExtractImagePatches Grad DynamicShape.
    Description: Test case of dynamic rank for ExtractImagePatches grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    extract_image_patches_test(True)
