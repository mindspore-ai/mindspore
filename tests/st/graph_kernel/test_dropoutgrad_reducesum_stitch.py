# Copyright 2021 Huawei Technologies Co., Ltd
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
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as GP
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark


class BertAttentionGradPiece(Cell):
    def __init__(self):
        super(BertAttentionGradPiece, self).__init__()
        self.add = P.Add()
        self.reducesum = P.ReduceSum(keep_dims=True)
        self.dropout_grad = GP.DropoutGrad(1 - 0.1)
        self.sub = P.Sub()
        self.multiply = P.Mul()
        self.cast = P.Cast()

    def construct(self, x, y, z):
        out1 = self.dropout_grad(x, y)
        out2 = self.multiply(out1, z)
        out3 = self.reducesum(self.cast(out2, mstype.float32), (-1,))
        out4 = self.sub(out1, self.cast(out3, mstype.float16))
        return out4


def get_rtol_atol(dtype):
    if dtype == np.float16:
        return 1.e-3, 1.e-3
    return 1.e-4, 1.e-4


def compare_result(expect, output, dtype):
    rtol, atol = get_rtol_atol(dtype)
    if isinstance(expect, (list, tuple)):
        assert isinstance(output, (list, tuple)) and len(expect) == len(output)
        expect_list = list(expect)
        output_list = list(output)
        for e, o in zip(expect_list, output_list):
            assert np.allclose(e.asnumpy(), o.asnumpy(), rtol, atol, equal_nan=True)
    else:
        assert np.allclose(expect.asnumpy(), output.asnumpy(), rtol, atol, equal_nan=True)


def get_dropoutgrad_reducesum_output(x, y, z, enable_stitch_fusion):
    # enable graph kernel stitch fusion.
    if enable_stitch_fusion:
        context.set_context(graph_kernel_flags="--enable_stitch_fusion=true")
    net = BertAttentionGradPiece()
    result = net(x, y, z)
    return result


def run_dropoutgrad_reducesum(shape, dtype):
    np.random.seed(0)
    x = Tensor(np.random.normal(0, 1, shape).astype(dtype))
    y = Tensor(np.random.normal(0, 1, shape).astype(dtype))
    z = Tensor(np.random.normal(0, 1, shape).astype(dtype))
    expect = get_dropoutgrad_reducesum_output(x, y, z, False)
    output = get_dropoutgrad_reducesum_output(x, y, z, True)
    compare_result(expect, output, dtype)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dropoutgrad_reducesum_gpu():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(enable_graph_kernel=True)
    context.set_context(mode=context.GRAPH_MODE)
    run_dropoutgrad_reducesum([64, 12, 128, 128], np.float16)
