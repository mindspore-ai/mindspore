# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
import mindspore.ops.operations._grad_ops as P
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
np.random.seed(1)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_atangrad_fp32():
    x_np = np.random.rand(4, 2).astype(np.float32) * 10
    dout_np = np.random.rand(4, 2).astype(np.float32) * 10
    output_ms = P.AtanGrad()(Tensor(x_np), Tensor(dout_np))
    output_np = dout_np / (1 + np.square(x_np))
    assert np.allclose(output_ms.asnumpy(), output_np, 1e-4, 1e-4)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_atangrad_fp16():
    x_np = np.random.rand(4, 2).astype(np.float16) * 10
    dout_np = np.random.rand(4, 2).astype(np.float16) * 10
    output_ms = P.AtanGrad()(Tensor(x_np), Tensor(dout_np))
    output_np = dout_np.astype(np.float32) / (1 + np.square(x_np.astype(np.float32)))
    assert np.allclose(output_ms.asnumpy(), output_np.astype(np.float16), 1e-3, 1e-3)
