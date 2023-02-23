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
import pytest

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
# enable graph kernel optimization.
context.set_context(enable_graph_kernel=True)


class EmbeddingPostprocessor(Cell):
    def __init__(self):
        super(EmbeddingPostprocessor, self).__init__()
        self.layernorm = nn.LayerNorm((768,))
        self.add = P.Add()
        self.dropout = nn.Dropout(p=0.1)

    def construct(self, word_embeddings, token_type_embeddings, position_embeddings):
        output = word_embeddings
        output = self.add(output, token_type_embeddings)
        output = self.add(output, position_embeddings)
        output = self.layernorm(output)
        output = self.dropout(output)
        return output


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


def get_layernorm_output(x, y, z, enable_stitch_fusion):
    # enable graph kernel stitch fusion.
    if enable_stitch_fusion:
        context.set_context(graph_kernel_flags="--enable_stitch_fusion=true")
    net = EmbeddingPostprocessor()
    result = net(x, y, z)
    return result


def test_layernorm(shape1, shape2, dtype):
    np.random.seed(0)
    x = Tensor(np.random.normal(0, 1, shape1).astype(dtype))
    y = Tensor(np.random.normal(0, 1, shape1).astype(dtype))
    z = Tensor(np.random.normal(0, 1, shape2).astype(dtype))
    expect = get_layernorm_output(x, y, z, False)
    output = get_layernorm_output(x, y, z, True)
    compare_result(expect, output, dtype)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_layernorm_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_layernorm([8192, 768], [1, 768], np.float32)
