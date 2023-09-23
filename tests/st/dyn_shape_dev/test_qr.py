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
import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops


@ms.jit
def qr_forward_func(x, full_matrices):
    return ops.auto_generate.qr_(x, full_matrices)


@ms.jit
def qr_backward_func(x, full_matrices):
    return ops.grad(qr_forward_func, (0, 1))(x, full_matrices)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_qr_forward(mode):
    """
    Feature: qr ops.
    Description: test ops qr.
    Expectation: output right results.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[20., -31, 7],
                         [4, 270, -90],
                         [-8, 17, -32]]).astype(np.float32))
    output_q, output_r = qr_forward_func(x, False)
    print("output_q:\n", output_q)
    print("output_r:\n", output_r)
    expect_output_q = np.asarray([[-0.912871, 0.16366126, 0.37400758],
                                  [-0.18257418, -0.9830709, -0.01544376],
                                  [0.36514837, -0.08238228, 0.92729706]]).astype(np.float32)
    expect_output_r = np.asarray([[-21.908903, -14.788506, -1.6431675],
                                  [0., -271.9031, 92.25824],
                                  [0., 0., -25.665514]]).astype(np.float32)
    assert np.allclose(output_q.asnumpy(), expect_output_q)
    assert np.allclose(output_r.asnumpy(), expect_output_r)
