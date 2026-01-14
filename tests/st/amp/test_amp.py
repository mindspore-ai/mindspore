# Copyright 2025 Huawei Technologies Co., Ltd
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
from mindspore import ops, nn, Parameter, Tensor, context
from mindspore.train.amp import auto_mixed_precision
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_amp_white_black_list(mode):
    """
    Feature: auto mixed precision auto mode.
    Description: input(fp32) -> MatMul(WhiteList) ->Tan(BlackList)
    Expectation: out1 fp16, out2 fp32.
    """
    context.set_context(mode=mode)
    class WhiteBlackNet(nn.Cell):
        def __init__(self, weight_shape, input_type_np=np.float32):
            super().__init__()
            self.weight = Parameter(Tensor(np.random.randn(*weight_shape).astype(input_type_np)),
                                    name="weight")
            self.matmul = ops.MatMul()
            self.tan = ops.Tan()

        def construct(self, x):
            out1 = self.matmul(x, self.weight)
            out2 = self.tan(out1)
            return out1, out2
    input_x = Tensor(np.random.randn(16, 16).astype(np.float32))
    net = auto_mixed_precision(WhiteBlackNet((16, 16)), amp_level="auto")
    out1, out2 = net(input_x)
    assert out1.dtype == ms.float16
    assert out2.dtype == ms.float32


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_amp_white_auto_promote(mode):
    """
    Feature: auto mixed precision auto mode.
    Description: input(fp32) -> MatMul(WhiteList) ->BiasAdd(AutoPromoteLis)
    Expectation: out1 fp16, out2 fp32.
    """
    context.set_context(mode=mode)
    class WhiteAutoPromoteNet(nn.Cell):
        def __init__(self, weight_shape, bias_shape, input_type_np=np.float32, dtype=ms.float16):
            super().__init__()
            self.weight = Parameter(Tensor(np.random.randn(*weight_shape).astype(input_type_np)),
                                    name="weight")
            self.bias = Tensor(np.random.randn(*bias_shape), dtype=dtype)
            self.matmul = ops.MatMul()
            self.biasadd = ops.BiasAdd()

        def construct(self, x):
            out1 = self.matmul(x, self.weight)
            out2 = self.biasadd(out1, self.bias)  # promote: fp16+fp16=fp16
            return out1, out2

    input_x = Tensor(np.random.randn(16, 16).astype(np.float32))
    net = auto_mixed_precision(WhiteAutoPromoteNet((16, 16), (16,)), amp_level="auto", dtype=ms.bfloat16)
    out1, out2 = net(input_x)
    assert out1.dtype == ms.bfloat16
    assert out2.dtype == ms.float32
