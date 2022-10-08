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

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.ops import composite as C
from mindspore import Tensor
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class MaxPool(nn.Cell):
    def __init__(self, kernel_size, strides, pad_mode):
        super().__init__()
        self.maxpool = P.MaxPool(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)

    def construct(self, x):
        return self.maxpool(x)


class MaxPoolGrad(nn.Cell):
    def __init__(self, forward):
        super().__init__()
        self.forward = forward
        self.grad = C.GradOperation(get_all=True, sens_param=True)

    def construct(self, x, sens):
        return self.grad(self.forward)(x, sens)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maxpool_grad_vmap():
    """
    Feature: test MaxPoolGrad vmap feature.
    Description: test MaxPoolGrad vmap feature.
    Expectation: success.
    """
    in_axes = -1
    seed = np.random.RandomState()
    x = Tensor(seed.random((1, 1, 6, 6, 3, 6)).astype(np.float32))
    sens = Tensor(seed.random((1, 1, 3, 3, 3, 6)).astype(np.float32))
    maxpool = MaxPool(kernel_size=2, strides=2, pad_mode="VALID")
    bp = MaxPoolGrad(maxpool)
    maxpoolgrad_vmap = vmap(vmap(bp, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = maxpoolgrad_vmap(x, sens)

    assert out[0].shape == (6, 3, 1, 1, 6, 6)
