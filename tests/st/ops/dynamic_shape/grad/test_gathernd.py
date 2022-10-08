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


class GatherNdNet(nn.Cell):
    def __init__(self):
        super(GatherNdNet, self).__init__()
        self.gathernd = ops.GatherNd()

    def construct(self, x, indices):
        return self.gathernd(x, indices)


def gathernd_test(is_dyn_rank):
    x = Tensor(np.random.randn(3, 2).astype(np.float32))
    indices = Tensor(np.array([[1, 1], [0, 1]]).astype(np.int32))
    tester = TestDynamicGrad(GatherNdNet(), skip_convert_out_ids=[0])
    tester.test_dynamic_grad_net([x, indices], is_dyn_rank)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_gathernd_dyn_shape():
    """
    Feature: GatherND Grad DynamicShape.
    Description: Test case of dynamic shape for GatherND grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    gathernd_test(False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_gathernd_dyn_rank():
    """
    Feature: GatherND Grad DynamicShape.
    Description: Test case of dynamic rank for GatherND grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    gathernd_test(True)
