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
from mindspore.ops import split, interpolate
from mindspore import ops
from mindspore.ops.auto_generate.gen_pyboost_func import baddbmm, transpose, view, bmm, exp, erf, silu, sin, cos, \
    cast, add, sub, softmax, sqrt, stack, split_tensor, split_with_size, matmul, conv2d, gather, broadcast_to, \
    maximum, minimum, greater_equal, less, unsqueeze, masked_fill, layer_norm, mean, cat, square
from mindspore.ops.auto_generate.gen_pyboost_func import pow as pyboost_pow
from mindspore.ops.auto_generate.gen_pyboost_func import sum as pyboost_sum
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
def test_add_ascend():
    """
    Feature: test add_ext operator
    Description: test add_ext run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    x = Tensor(np.array([[[1, 3, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
    y = Tensor(np.array([[[1, 3, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
    output = add(x, y, 1)
    assert np.allclose(output.asnumpy(), [[[2, 6, 6], [8, 10, 12]], [[14, 16, 18], [20, 22, 24]]])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sub_ascend():
    """
    Feature: test add_ext operator
    Description: test sub_ext run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    x = Tensor(np.array([[[1, 3, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
    y = Tensor(np.array([[[1, 1, 1], [1, 1, 2]], [[3, 8, 9], [10, 11, 12]]]), mindspore.float32)
    output = sub(x, y, 1)
    assert np.allclose(output.asnumpy(), [[[0, 2, 2], [3, 4, 4]], [[4, 0, 0], [0, 0, 0]]])


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


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_split_tensor_ascend():
    """
    Feature: test split_tensor operator
    Description: test split_tensor run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x = np.arange(9).astype("float32")
    outputs = split_tensor(Tensor(input_x), 3)
    assert np.allclose(outputs[0].asnumpy(), [0, 1, 2])
    assert np.allclose(outputs[1].asnumpy(), [3, 4, 5])
    assert np.allclose(outputs[2].asnumpy(), [6, 7, 8])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_softmax_ascend():
    """
    Feature: test cast operator
    Description: test softmax run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float32)
    output = softmax(x, 0)
    assert np.allclose(output.asnumpy(), [0.03168492, 0.01165623, 0.08612854, 0.6364086, 0.23412167])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_split_with_size_ascend():
    """
    Feature: test split_with_size operator
    Description: test split_with_size run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x = np.arange(9).astype("float32")
    outputs = split_with_size(Tensor(input_x), (3, 3, 3))
    assert np.allclose(outputs[0].asnumpy(), [0, 1, 2])
    assert np.allclose(outputs[1].asnumpy(), [3, 4, 5])
    assert np.allclose(outputs[2].asnumpy(), [6, 7, 8])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sqrt_ascend():
    """
    Feature: test cast operator
    Description: test sqrt run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    x = Tensor(np.array([1.0, 4.0, 9.0]), mindspore.float32)
    output = sqrt(x)
    assert np.allclose(output.asnumpy(), [1, 2, 3])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_split_ascend():
    """
    Feature: test split operator
    Description: test split run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x = np.arange(9).astype("float32")
    outputs = split(Tensor(input_x), (3, 3, 3))
    assert np.allclose(outputs[0].asnumpy(), [0, 1, 2])
    assert np.allclose(outputs[1].asnumpy(), [3, 4, 5])
    assert np.allclose(outputs[2].asnumpy(), [6, 7, 8])
    outputs = split(Tensor(input_x), 3)
    assert np.allclose(outputs[0].asnumpy(), [0, 1, 2])
    assert np.allclose(outputs[1].asnumpy(), [3, 4, 5])
    assert np.allclose(outputs[2].asnumpy(), [6, 7, 8])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_stack_ascend():
    """
    Feature: test cast operator
    Description: test stack run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([0, 1]).astype(np.float32))
    input_x2 = Tensor(np.array([2, 3]).astype(np.float32))
    output = stack((input_x1, input_x2), 0)
    assert np.allclose(output.asnumpy(), [[0, 1], [2, 3]])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pow_ascend():
    """
    Feature: test cast operator
    Description: test pow run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([1, 2]).astype(np.float32))
    input_x2 = Tensor(np.array([2, 3]).astype(np.float32))
    output = pyboost_pow(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), [1, 8])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_ascend():
    """
    Feature: test cast operator
    Description: test matmul run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    input_x2 = Tensor(np.array([[2, 3], [4, 5]]).astype(np.float32))
    output = matmul(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), [[10, 13], [22, 29]])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_conv2d_ascend():
    """
    Feature: test conv2d_ext operator
    Description: test conv2d_ext run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")

    np.random.seed(1)
    filters = Tensor(np.random.randn(8, 4, 3, 3).astype(dtype=np.float32))
    inputs = Tensor(np.random.randn(1, 4, 5, 5).astype(dtype=np.float32))
    bias = Tensor(np.random.randn(8).astype(dtype=np.float32))

    pyboost_out = conv2d(inputs, filters, padding=1, bias=bias)
    old_out = ops.conv2d(inputs, filters, pad_mode="pad", padding=1, bias=bias)
    assert np.allclose(pyboost_out.asnumpy(), old_out.asnumpy(), 0.003, 0.003)
    pyboost_out = conv2d(inputs, filters, padding=1)
    old_out = ops.conv2d(inputs, filters, pad_mode="pad", padding=1)
    assert np.allclose(pyboost_out.asnumpy(), old_out.asnumpy(), 0.003, 0.003)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_gather_ascend():
    """
    Feature: test cast operator
    Description: test gather run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([0, 1, 2, 3, 4, 5, 6, 7]).astype(np.float32))
    input_x2 = Tensor(np.array([0, 1, 2, 3, 3, 2, 1, 0]).astype(np.int64))
    output = gather(input_x1, 0, input_x2)
    assert np.allclose(output.asnumpy(), [0, 1, 2, 3, 3, 2, 1, 0])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_broadcast_to_ascend():
    """
    Feature: test cast operator
    Description: test broadcast_to run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([1, 2, 3]).astype(np.float32))
    input_x2 = [2, 3]
    output = broadcast_to(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), [[1, 2, 3], [1, 2, 3]])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_maximum_ascend():
    """
    Feature: test cast operator
    Description: test pow run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([1, 2, -1]).astype(np.float32))
    input_x2 = Tensor(np.array([3, 0, 4]).astype(np.float32))
    output = maximum(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), [3, 2, 4])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_minimum_ascend():
    """
    Feature: test cast operator
    Description: test pow run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([1, 2, -1]).astype(np.float32))
    input_x2 = Tensor(np.array([3, 0, 4]).astype(np.float32))
    output = minimum(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), [1, 0, -1])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_greater_equal_ascend():
    """
    Feature: test cast operator
    Description: test pow run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    input_x2 = Tensor(np.array([[1, 1], [4, 4]]).astype(np.float32))
    output = greater_equal(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), [[True, True], [False, True]])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_less_ascend():
    """
    Feature: test cast operator
    Description: test pow run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    input_x2 = Tensor(np.array([[1, 1], [4, 4]]).astype(np.float32))
    output = less(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), [[False, False], [True, False]])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_unsqueeze_ascend():
    """
    Feature: test cast operator
    Description: test unsqueeze run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([1, 2, 3, 4]).astype(np.float32))
    input_x2 = 1
    output = unsqueeze(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), [[1], [2], [3], [4]])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_masked_fill_ascend():
    """
    Feature: test cast operator
    Description: test masked_fill run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([1, 2, 3, 4]).astype(np.float32))
    input_x2 = Tensor(np.array([True, True, False, True]), mindspore.bool_)
    output = masked_fill(input_x1, input_x2, 0.5)
    assert np.allclose(output.asnumpy(), [0.5, 0.5, 3, 0.5])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_layer_norm_ascend():
    """
    Feature: test cast operator
    Description: test layer_norm run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([[[1, 2], [3, 4]]]).astype(np.float32))
    weight = Tensor(np.ones(2), mindspore.float32)
    bias = Tensor(np.zeros(2), mindspore.float32)
    output = layer_norm(input_x1, [2], weight, bias, 0.00001)
    assert np.allclose(output[0].asnumpy(), [[[-1.0, 1.0], [-1.0, 1.0]]])
    assert np.allclose(output[1].asnumpy(), [[[1.5], [3.5]]])
    assert np.allclose(output[2].asnumpy(), [[[2.0], [2.0]]])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_implicit_cast_ascend():
    """
    Feature: test implicit cast operator
    Description: test implicit cast run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    x = Tensor(np.array([2]))
    y = Tensor(np.array([3]), mindspore.float32)
    out = add(x, y)
    assert np.allclose(out.asnumpy(), np.array([5], np.float32))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sum_ascend():
    """
    Feature: test cast operator
    Description: test sum run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([1, 2, 3, 4]).astype(np.float32))
    output = pyboost_sum(input_x1, 0, False, mindspore.float32)
    assert np.allclose(output.asnumpy(), [10])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mean_ascend():
    """
    Feature: test cast operator
    Description: test mean run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([1, 2, 3, 4]).astype(np.float32))
    output = mean(input_x1, 0, False, mindspore.float32)
    assert np.allclose(output.asnumpy(), [2.5])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_cat_ascend():
    """
    Feature: test cast operator
    Description: test cat run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([0, 1]).astype(np.float32))
    input_x2 = Tensor(np.array([2, 3]).astype(np.float32))
    output = cat((input_x1, input_x2), 0)
    assert np.allclose(output.asnumpy(), [0, 1, 2, 3])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_interpolate_ascend():
    """
    Feature: test interpolate operator
    Description: test interpolate run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")

    np.random.seed(1)
    x = Tensor(np.ones(shape=[1, 3, 3]), mindspore.float32)
    expect_out = Tensor(np.ones(shape=[1, 3, 5]), mindspore.float32)
    pyboost_out = interpolate(x, size=(5, ), mode='nearest')
    assert np.allclose(pyboost_out.asnumpy(), expect_out.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_square_ascend():
    """
    Feature: test cast operator
    Description: test square run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    input_x1 = Tensor(np.array([1, 2]).astype(np.float32))
    output = square(input_x1)
    assert np.allclose(output.asnumpy(), [1, 4])
