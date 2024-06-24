# Copyright 2024 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, jit, JitConfig
from mindspore.ops.auto_generate import FFNExt

class Net(nn.Cell):
    def __init__(self, activation_, innerPrecise_):
        super(Net, self).__init__()
        self.ffn = FFNExt(activation=activation_, inner_precise=innerPrecise_)

    def construct(self, x_, w1_, w2_, expert_tokens_, bias1_, bias2_):
        return self.ffn(x_, w1_, w2_, expert_tokens_, bias1_, bias2_)

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def ffn_forward_func(x_, w1_, w2_, expert_tokens_, bias1_, bias2_):
    ffn = FFNExt("fastgelu", 1)
    return ffn(x_, w1_, w2_, expert_tokens_, bias1_, bias2_)

x = Tensor(np.array([[0.2062, 0.9795, 0.02841, 0.438],
                     [0.776, 0.56, 0.6953, 0.8276],
                     [0.874, 0.4092, 0.507, 0.8447],
                     [0.1992, 0.6846, 0.3013, 0.973]], dtype=np.float16))

w1 = Tensor(np.array([[[0.917, 0.2856, 0.873, 0.3623, 0.718, 0.779, 0.6245,
                        0.918, 0.1174, 0.3323, 0.3928, 0.532, 0.1394, 0.7563,
                        0.3552, 0.6455],
                       [0.5933, 0.4075, 0.9688, 0.96, 0.53, 0.752, 0.875,
                        0.09235, 0.5044, 0.6025, 0.95, 0.2382, 0.46, 0.5156,
                        0.6743, 0.02383],
                       [0.9517, 0.314, 0.1627, 0.7407, 0.2808, 0.919, 0.8477,
                        0.2462, 0.7876, 0.7383, 0.4644, 0.91, 0.4055, 0.291,
                        0.8945, 0.3154],
                       [0.2659, 0.447, 0.9585, 0.183, 0.1794, 0.811, 0.2698,
                        0.2854, 0.713, 0.389, 0.557, 0.0771, 0.706, 0.2235,
                        0.6274, 0.7603]],
                      [[0.4478, 0.622, 0.2493, 0.01236, 0.3071, 0.3096, 0.2324,
                        0.53, 0.728, 0.3242, 0.435, 0.591, 0.3345, 0.9736,
                        0.5933, 0.737],
                       [0.2048, 0.3484, 0.2073, 0.3823, 0.5195, 0.767, 0.449,
                        0.868, 0.756, 0.368, 0.9175, 0.452, 0.9897, 0.1078,
                        0.06415, 0.4026],
                       [0.1787, 0.6094, 0.907, 0.0378, 0.3018, 0.2078, 0.2072,
                        0.4534, 0.174, 0.08514, 0.4438, 0.6016, 0.4385, 0.5474,
                        0.8516, 0.4407],
                       [0.5537, 0.96, 0.781, 0.2179, 0.2957, 0.3318, 0.3906,
                        0.501, 0.2585, 0.4895, 0.2659, 0.00804, 0.4622, 0.2732,
                        0.571, 0.10785]]], dtype=np.float16))

w2 = Tensor(np.array([[[0.809, 0.1511, 0.01147, 0.3022],
                       [0.1381, 0.1565, 0.8384, 0.3955],
                       [0.5557, 0.9688, 0.11334, 0.568],
                       [0.599, 0.1973, 0.593, 0.9004],
                       [0.947, 0.0715, 0.1103, 0.7964],
                       [0.669, 0.8955, 0.486, 0.014305],
                       [0.4258, 0.1781, 0.399, 0.011696],
                       [0.1837, 0.2964, 0.298, 0.3506],
                       [0.688, 0.3086, 0.2139, 0.7896],
                       [0.6064, 0.956, 0.675, 0.941],
                       [0.2556, 0.2032, 0.711, 0.03397],
                       [0.6797, 0.982, 0.907, 0.4036],
                       [0.05685, 0.2084, 0.003544, 0.3184],
                       [0.396, 0.4917, 0.0804, 0.98],
                       [0.998, 0.1427, 0.655, 0.9434],
                       [0.8105, 0.7573, 0.7705, 0.645]],

                      [[0.339, 0.02399, 0.3816, 0.2937],
                       [0.3013, 0.3745, 0.1342, 0.507],
                       [0.536, 0.5317, 0.7437, 0.3433],
                       [0.185, 0.908, 0.958, 0.8047],
                       [0.5386, 0.01852, 0.0949, 0.9673],
                       [0.4622, 0.2798, 0.9526, 0.575],
                       [0.395, 0.12177, 0.5317, 0.0941],
                       [0.7803, 0.3743, 0.9385, 0.878],
                       [0.1328, 0.499, 0.1537, 0.542],
                       [0.359, 0.4695, 0.8794, 0.2866],
                       [0.631, 0.4766, 0.997, 0.764],
                       [0.206, 0.7075, 0.8154, 0.05804],
                       [0.4128, 0.6733, 0.05515, 0.8833],
                       [0.5107, 0.7393, 0.3132, 0.6865],
                       [0.3357, 0.677, 0.266, 0.464],
                       [0.8687, 0.9624, 0.646, 0.6953]]], dtype=np.float16))

expert_tokens = Tensor(np.array([2, 2]))

bias1 = Tensor(np.array([[0.678, 0.5293, 0.571, 0.932, 0.741, 0.0552, 0.1658,
                          0.07495, 0.665, 0.849, 0.2168, 0.964, 0.314, 0.823,
                          0.303, 0.1957],
                         [0.712, 0.769, 0.0927, 0.4043, 0.846, 0.3271, 0.637,
                          0.636, 0.2778, 0.0754, 0.876, 0.3218, 0.7085, 0.1785,
                          0.1526, 0.1835]], dtype=np.float16))

bias2 = Tensor(np.array([[0.9985, 0.5, 0.532, 0.6343],
                         [0.638, 0.979, 0.2174, 0.3445]], dtype=np.float16))

expect = np.array([[12.805, 9.766, 8.98, 12.16],
                   [19.38, 15.04, 13.734, 17.61],
                   [11.49, 12.05, 12.41, 14.17],
                   [9.82, 9.91, 10.79, 12.32]])

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ffn_forward_net():
    """
    Feature: Test moeffn in kbk and pynative mode.
    Description: call _inner_ops.FFN with valid input.
    Expectation: return the correct value.
    """

    net = Net('fastgelu', 1)
    output = net(x, w1, w2, expert_tokens, bias1, bias2)

    assert np.allclose(output.asnumpy(), expect, rtol=1e-1)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_ffn_forward_mode(mode):
    """
    Feature: Test ffn with static shape in kbk and pynative mode.
    Description: call kbk with valid input.
    Expectation: return the correct value.
    """

    if mode == 'pynative':
        output = ffn_forward_func(x, w1, w2, expert_tokens, bias1, bias2)
    elif mode == 'KBK':
        output = (jit(ffn_forward_func, jit_config=JitConfig(jit_level="O0")))(x, w1, w2, expert_tokens, bias1, bias2)
    else:
        output = (jit(ffn_forward_func, jit_config=JitConfig(jit_level="O2")))(x, w1, w2, expert_tokens, bias1, bias2)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-1)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ffn_forward_value():
    """
    Feature: Test moeffn in kbk and pynative mode.
    Description: call _inner_ops.FFN with valid input.
    Expectation: return the correct value.
    """

    output = ffn_forward_func(x, w1, w2, expert_tokens, bias1, bias2)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ffn_forward_shape():
    """
    Feature: Test moeffn in kbk and pynative mode.
    Description: call _inner_ops.FFN with valid input.
    Expectation: return the correct value.
    """

    x_ = Tensor(generate_random_input((4, 4), np.float16))
    w1_ = Tensor(generate_random_input((2, 4, 16), np.float16))
    w2_ = Tensor(generate_random_input((2, 16, 4), np.float16))
    expert_tokens_ = Tensor(np.full(2, 2))
    bias1_ = Tensor(generate_random_input((2, 16), np.float16))
    bias2_ = Tensor(generate_random_input((2, 4), np.float16))

    output = ffn_forward_func(x_, w1_, w2_, expert_tokens_, bias1_, bias2_)
    expect_shape = (4, 4)
    assert np.allclose(output.shape, expect_shape)
