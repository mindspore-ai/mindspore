# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Removing the ops wrapper must not change the independent BiDense layer."""
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('batch_shape', [(), (2,), (2, 3)])
@pytest.mark.parametrize('has_bias', [False, True])
def test_retained_bidense_values(mode, batch_shape, has_bias):
    """
    Feature: Retained independent nn.BiDense implementation.
    Description: Evaluate fixed weights, ranks and optional bias in graph and eager modes.
    Expectation: Output matches NumPy bilinear contraction, with unchanged shape and dtype.
    """
    previous = ms.get_context('mode')
    try:
        ms.set_context(mode=mode)
        x = np.arange(np.prod(batch_shape + (2,)), dtype=np.float32).reshape(batch_shape + (2,)) / 10
        y = np.arange(np.prod(batch_shape + (3,)), dtype=np.float32).reshape(batch_shape + (3,)) / 10
        weight = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 10
        bias = np.array([0.25, -0.5], dtype=np.float32)
        layer = nn.BiDense(2, 3, 2, weight_init=ms.Tensor(weight), bias_init=ms.Tensor(bias),
                           has_bias=has_bias)
        result = layer(ms.Tensor(x), ms.Tensor(y))
        expected = np.einsum('...i,oij,...j->...o', x, weight, y)
        if has_bias:
            expected = expected + bias
        np.testing.assert_allclose(result.asnumpy(), expected, rtol=1e-5, atol=1e-6)
        assert result.shape == batch_shape + (2,)
        assert result.dtype == ms.float32
    finally:
        ms.set_context(mode=previous)
