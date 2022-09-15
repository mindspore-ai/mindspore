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
from mindspore import ops, nn, context, Tensor
from .test_grad_of_dynamic import TestDynamicGrad


class Net(nn.Cell):
    def __init__(self, ksizes, strides, padding="valid"):
        super(Net, self).__init__()
        self.extractvolumepatches = ops.ExtractVolumePatches(ksizes, strides, padding)

    def construct(self, input_tensor):
        return self.extractvolumepatches(input_tensor)


def extract_volume_patches_test(is_dyn_rank):
    net = Net([1, 1, 2, 2, 2], [1, 1, 1, 1, 1], "VALID")
    input_tensor = Tensor(np.random.rand(1, 1, 3, 3, 3).astype(np.float16))
    tester = TestDynamicGrad(net)
    tester.test_dynamic_grad_net([input_tensor], is_dyn_rank)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_extract_volume_patches_dyn_shape():
    """
    Feature: ExtractVolumePatches Grad DynamicShape.
    Description: Test case of dynamic shape for ExtractVolumePatches grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    extract_volume_patches_test(False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_extract_volume_patches_dyn_rank():
    """
    Feature: ExtractvolumePatches Grad DynamicShape.
    Description: Test case of dynamic rank for ExtractVolumePatches grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    extract_volume_patches_test(True)
