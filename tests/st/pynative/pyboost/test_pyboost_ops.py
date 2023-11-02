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
from mindspore import Tensor, context
from mindspore import nn
from mindspore.ops.composite import GradOperation
from mindspore.ops.auto_generate import baddbmm, transpose, view, bmm, exp, erf, silu, sin, cos, cast, add_ext 
import mindspore


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_baddbmm_ascend():
    """
    Feature: test baddbmm operator
    Description: test baddbmm run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    op_input = Tensor(np.ones([1, 3, 3]).astype(np.float32))
    batch1 = Tensor(np.ones([1, 3, 4]).astype(np.float32))
    batch2 = Tensor(np.ones([1, 4, 3]).astype(np.float32))
    output = baddbmm(op_input, batch1, batch2, 1, 1)
    output = baddbmm(output, batch1, batch2, 1, 1)
    assert (output.asnumpy() == np.ones([1, 3, 3]).astype(np.float32) * 9).all()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
class Baddbmm(nn.Cell):
    def __init__(self):
        super(Baddbmm, self).__init__()
        self.input = 1

    def construct(self, op_input, batch1, batch2):
        return baddbmm(op_input, batch1, batch2, 1, 1)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_baddbmm_grad():
    """
    Feature: test baddbmm grad
    Description: test baddbmm grad run by pyboost
    Expectation: success
    """
    op_input = Tensor(np.ones([1, 3, 3]).astype(np.float32))
    batch1 = Tensor(np.ones([1, 3, 4]).astype(np.float32))
    batch2 = Tensor(np.ones([1, 4, 3]).astype(np.float32))
    network = Baddbmm()
    grad_op = GradOperation(get_all=True)(network)
    grad = grad_op(op_input, batch1, batch2)
    assert (grad[0].asnumpy() == np.ones([1, 3, 3]).astype(np.float32)).all()
    assert (grad[1].asnumpy() == np.ones([1, 3, 4]).astype(np.float32) * 3).all()
    assert (grad[2].asnumpy() == np.ones([1, 4, 3]).astype(np.float32) * 3).all()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_transpose_ascend():
    """
    Feature: test transpose operator
    Description: test transpose run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")

    op_input = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
    input_perm = (0, 2, 1)
    output = transpose(op_input, input_perm)
    assert np.allclose(output.asnumpy(), [[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_view_ascend():
    """
    Feature: test view operator
    Description: test view run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")

    op_input = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
    output = view(op_input, (3, 2))
    assert np.allclose(output.asnumpy(), [[-0.1, 0.3], [3.6, 0.4], [0.5, -3.2]])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_bmm_ascend():
    """
    Feature: test bmm operator
    Description: test bmm run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    op_input = Tensor(np.ones([1, 3, 4]).astype(np.float32))
    mat2 = Tensor(np.ones([1, 4, 3]).astype(np.float32))
    output = bmm(op_input, mat2)
    output = bmm(mat2, output)
    except_data = np.ones([1, 4, 3]).astype(np.float32) * 12
    assert (output.asnumpy() == except_data).all()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_exp_ascend():
    """
    Feature: test exp operator
    Description: test exp run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")

    x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
    output = exp(x)
    assert np.allclose(output.asnumpy(), [2.718282, 7.389056, 54.598152])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_erf_ascend():
    """
    Feature: test erf operator
    Description: test erf run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")

    x = Tensor(np.array([-1, 0, 1, 2, 3]), mindspore.float32)
    output = erf(x)
    assert np.allclose(output.asnumpy(), [-0.8427168, 0., 0.8427168, 0.99530876, 0.99997765])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_silu_ascend():
    """
    Feature: test silu operator
    Description: test silu run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    x = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float32)
    output = silu(x)
    assert np.allclose(output.asnumpy(), [-0.26894143, 1.761594, -0.14227761, 1.761594, -0.26894143])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_add_ext_ascend():
    """
    Feature: test add_ext operator
    Description: test add_ext run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    x = Tensor(np.array([[[1, 3, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
    y = Tensor(np.array([[[1, 3, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
    output = add_ext(x, y)
    assert np.allclose(output.asnumpy(), [[[2, 6, 6], [8, 10, 12]], [[14, 16, 18], [20, 22, 24]]])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sin_ascend():
    """
    Feature: test sin operator
    Description: test sin run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
    output = sin(x)
    assert np.allclose(output.asnumpy(), [0.5810352, 0.27635565, 0.41687083, 0.5810352])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_cos_ascend():
    """
    Feature: test cos operator
    Description: test cos run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    self = Tensor(np.arange(5), mindspore.float32)
    out = cos(self)
    assert np.allclose(out.asnumpy(), [1, 0.5403023, -0.41614684, -0.9899925, -0.6536436])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_cast_ascend():
    """
    Feature: test cast operator
    Description: test cast run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    self = Tensor(np.arange(5), mindspore.float32)
    dtype_dist = mindspore.int32
    out = cast(self, dtype_dist)
    assert np.allclose(out.asnumpy(), self.astype(np.int32).asnumpy())
