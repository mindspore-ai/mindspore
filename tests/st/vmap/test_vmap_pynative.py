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
"""test vmap in pynative mode"""

import pytest
import numpy as np
import mindspore.context as context
import mindspore.ops.functional as F
from mindspore import dtype as mstype
from mindspore.common import Tensor
from mindspore.ops.functional import vmap
from mindspore.common.api import jit

context.set_context(mode=context.PYNATIVE_MODE)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_nested():
    """
    Feature: vmap
    Description: This case mainly tests the following `vmap` application scenarios in PyNative mode:
        1.Calling nested `vmap` functions.
        2.`fn` is a function wrapped `jit`.
        3.Function contains free variables.
    Expectation: success
    """
    outter_tensor = Tensor([1], mstype.float32)

    def add_fn(x):
        return F.add(x, outter_tensor)

    @jit
    def inner_vmap_fn(x, outter_tensor):
        vmap_funtion = vmap(add_fn, 1)
        out = vmap_funtion(x)
        output = out + outter_tensor
        return output

    def outter_vmap_fn(x):
        output = vmap(inner_vmap_fn, (0, None), 1)(x, outter_tensor)
        return output

    x_hat = Tensor([[[1., 2., 3.], [4., 5., 6.]],
                    [[2., 3., 4.], [5., 6., 7.]],
                    [[3., 4., 5.], [6., 7., 8.]],
                    [[4., 5., 6.], [7., 8., 9.]]], mstype.float32)

    result = outter_vmap_fn(x_hat)
    expect_result = Tensor([[[3., 6.], [4., 7.], [5., 8.], [6., 9.]],
                            [[4., 7.], [5., 8.], [6., 9.], [7., 10.]],
                            [[5., 8.], [6., 9.], [7., 10.], [8., 11.]]], mstype.float32)
    assert np.allclose(result.asnumpy(), expect_result.asnumpy())
