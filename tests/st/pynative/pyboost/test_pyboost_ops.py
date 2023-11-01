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

import numpy as np
from mindspore import Tensor, ops, context
from mindspore import nn
from mindspore import context
from mindspore.ops.composite import GradOperation
from mindspore.ops.auto_generate import baddbmm, transpose, view, bmm, exp, erf, silu, sin, cos
import mindspore


def test_baddbmm_ascend():
    context.set_context(device_target="Ascend")
    input = Tensor(np.ones([1, 3, 3]).astype(np.float32))
    batch1 = Tensor(np.ones([1, 3, 4]).astype(np.float32))
    batch2 = Tensor(np.ones([1, 4, 3]).astype(np.float32))
    output = baddbmm(input, batch1, batch2, 1, 1)
    output = baddbmm(output, batch1, batch2, 1, 1)
    assert (output.asnumpy() == np.ones([1, 3, 3]).astype(np.float32) * 9).all()


class Baddbmm(nn.Cell):
    def __init__(self):
        super(Baddbmm, self).__init__()

    def construct(self, input, batch1, batch2):
        return baddbmm(input, batch1, batch2, 1, 1)


def test_baddbmm_grad():
    input = Tensor(np.ones([1, 3, 3]).astype(np.float32))
    batch1 = Tensor(np.ones([1, 3, 4]).astype(np.float32))
    batch2 = Tensor(np.ones([1, 4, 3]).astype(np.float32))
    network = Baddbmm()
    grad_op = GradOperation(get_all=True)(network)
    grad = grad_op(input, batch1, batch2)
    assert (grad[0].asnumpy() == np.ones([1, 3, 3]).astype(np.float32)).all()
    assert (grad[1].asnumpy() == np.ones([1, 3, 4]).astype(np.float32) * 3).all()
    assert (grad[2].asnumpy() == np.ones([1, 4, 3]).astype(np.float32) * 3).all()


def test_transpose_ascend():
    context.set_context(device_target="Ascend")

    input = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
    input_perm = (0, 2, 1)
    output = transpose(input, input_perm)
    assert np.allclose(output.asnumpy(), [[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]])


def test_view_ascend():
    context.set_context(device_target="Ascend")

    input = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
    output = view(input, (3, 2))
    assert np.allclose(output.asnumpy(), [[-0.1, 0.3], [3.6, 0.4], [0.5, -3.2]])


def test_bmm_ascend():
    context.set_context(device_target="Ascend")
    input = Tensor(np.ones([1, 3, 4]).astype(np.float32))
    mat2 = Tensor(np.ones([1, 4, 3]).astype(np.float32))
    output = bmm(input, mat2)
    output = bmm(mat2, output)
    except_data = np.ones([1, 4, 3]).astype(np.float32) * 12
    assert (output.asnumpy() == except_data).all()


def test_exp_ascend():
    context.set_context(device_target="Ascend")

    x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
    output = exp(x)
    assert np.allclose(output.asnumpy(), [2.718282, 7.389056, 54.598152])


def test_erf_ascend():
    context.set_context(device_target="Ascend")

    x = Tensor(np.array([-1, 0, 1, 2, 3]), mindspore.float32)
    output = erf(x)
    assert np.allclose(output.asnumpy(), [-0.8427168, 0., 0.8427168, 0.99530876, 0.99997765])


def test_silu_ascend():
    context.set_context(device_target="Ascend")
    x = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float32)
    output = silu(x)
    assert np.allclose(output.asnumpy(), [-0.26894143, 1.761594, -0.14227761, 1.761594, -0.26894143])


def test_add_ascend():
    context.set_context(device_target="Ascend")
    x = Tensor(np.array([[[1, 3, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
    y = Tensor(np.array([[[1, 3, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
    output = add_ext(x, y)
    assert np.allclose(output.asnumpy(), [[[2, 6, 6], [8, 10, 12]], [[14, 16, 18], [20, 22, 24]]])


def test_sin_ascend():
    context.set_context(device_target="Ascend")
    x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
    output = sin(x)
    assert np.allclose(output.asnumpy(), [0.5810352, 0.27635565, 0.41687083, 0.5810352])


def test_cos_ascend():
    context.set_context(device_target="Ascend")
    self = Tensor(np.arange(5), mindspore.float32)
    out = cos(self)
    assert np.allclose(out.asnumpy(), [1, 0.5403023, -0.41614684, -0.9899925, -0.6536436])
