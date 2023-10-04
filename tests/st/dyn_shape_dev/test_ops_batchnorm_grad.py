# Copyright 2023 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
from mindspore import Tensor, context, Parameter
from mindspore.ops import auto_generate as P
import test_utils


@test_utils.run_with_cell
def batchnorm_grad_forward_func(dout, x, scale, mean, variance, reserve, is_training):
    return P.BatchNormGrad(is_training=is_training,
                           epsilon=1e-5,
                           data_format="NCHW")(dout, x, scale, mean, variance, reserve)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("device", ["GPU", "CPU"])
def test_bn_grad_op(is_training, data_type, mode, device):
    """
    Feature: BatchNorm cpu/gpu kernel
    Description: test default attr
    Expectation: match to np benchmark.
    """
    dout = Tensor(np.random.rand(10, 36, 12, 12).astype(data_type))
    x = Tensor(np.random.rand(10, 36, 12, 12).astype(data_type))
    scale = Tensor(np.random.rand(36).astype(data_type))
    mean = Tensor(np.random.rand(36).astype(data_type))
    variance = Tensor(np.random.rand(36).astype(data_type))
    reserve = Tensor(np.random.rand(36).astype(data_type))
    if is_training:
        scale = Parameter(scale)
        mean = Parameter(mean)
        variance = Parameter(variance)
        reserve = Parameter(reserve)
    context.set_context(mode=mode, device_target=device, precompile_only=True)
    output = batchnorm_grad_forward_func(dout, x, scale, mean, variance, reserve, is_training)
    print(output)
    assert output is None
