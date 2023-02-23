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
import mindspore.nn as nn
from mindspore.nn import Cell
from mindspore.ops import operations as P
import mindspore.ops.functional as F
import pytest

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
# enable graph kernel optimization.
context.set_context(enable_graph_kernel=True)


class BertAttentionPiece(Cell):
    def __init__(self):
        super(BertAttentionPiece, self).__init__()
        self.add = P.Add()
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax()
        self.multiply_data = -10000.0
        self.sub = P.Sub()
        self.multiply = P.Mul()
        self.get_dtype = P.DType()
        self.cast = P.Cast()

    def construct(self, attention_mask, attention_scores):
        multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                self.cast(attention_mask, self.get_dtype(attention_scores)))
        adder = self.multiply(multiply_out, self.multiply_data)
        attention_scores = self.add(adder, attention_scores)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)
        return attention_probs


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


def get_softmax_output(x, y, enable_stitch_fusion):
    # enable graph kernel stitch fusion.
    if enable_stitch_fusion:
        context.set_context(graph_kernel_flags="--enable_stitch_fusion=true")
    net = BertAttentionPiece()
    result = net(x, y)
    return result


def test_softmax(shape, dtype):
    np.random.seed(0)
    x = Tensor(np.random.normal(0, 1, shape).astype(dtype))
    y = Tensor(np.random.normal(0, 1, shape).astype(dtype))
    expect = get_softmax_output(x, y, False)
    output = get_softmax_output(x, y, True)
    compare_result(expect, output, dtype)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_softmax_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_softmax([64, 12, 128, 128], np.float16)
