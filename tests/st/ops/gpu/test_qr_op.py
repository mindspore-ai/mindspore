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
from mindspore import Tensor
import mindspore.context as context
from mindspore.common import dtype as mstype
import mindspore.ops.operations.math_ops as P


def my_method(input_x, full_matrices):
    qr_op = P.Qr(full_matrices=full_matrices)
    out = qr_op(Tensor(input_x))
    res = [out[0].asnumpy(), out[1].asnumpy()]
    return res


def qr_cmp(input_x, full_matrices):
    out_me = my_method(input_x, full_matrices)
    _out_q = Tensor([[-0.857143, 0.394286, 0.331429],
                     [-0.428571, -0.902857, -0.034286],
                     [0.285714, -0.171429, 0.942857]],
                    dtype=mstype.float32).asnumpy()
    _out_r = Tensor([[-14.000001, -21.00001, 14],
                     [0, -175, 70.000015],
                     [0, 0, -34.999996]],
                    dtype=mstype.float32).asnumpy()
    np.testing.assert_allclose(out_me[0], _out_q, rtol=1e-3)
    np.testing.assert_allclose(out_me[1], _out_r, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_qr_pynative():
    """
    Feature: Qr_pynative
    Description: test cases for qr: m >= n and full_matrices=True
    Expectation: the result match to tf
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.array([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
    qr_cmp(input_x=x, full_matrices=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_qr_graph():
    """
    Feature: Qr_graph
    Description: test cases for qr: m < n and full_matrices=False
    Expectation: the result match to tf
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
    qr_cmp(input_x=x, full_matrices=False)
