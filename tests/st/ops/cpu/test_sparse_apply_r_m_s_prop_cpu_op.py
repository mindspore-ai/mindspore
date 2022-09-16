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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops.operations as P
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class SparseApplyRMSPropNet(nn.Cell):
    def __init__(self, rho, momentum, epsilon, use_locking=False):
        super(SparseApplyRMSPropNet, self).__init__()
        self.sparse_apply_r_m_s_prop = P.SparseApplyRMSProp(rho, momentum, epsilon, use_locking)
        self.var = Parameter(Tensor(np.array([[0.6, 0.3], [0.1, 0.5]]).astype(np.float32)), name="var")
        self.ms = Parameter(Tensor(np.array([[0.2, 0.4], [0.1, 0.3]]).astype(np.float32)), name="ms")
        self.mom = Parameter(Tensor(np.array([[0.3, 0.1], [0.3, 0.6]]).astype(np.float32)), name="mom")

    def construct(self, learning_rate, grad, indices):
        out = self.sparse_apply_r_m_s_prop(self.var, self.ms, self.mom, learning_rate, grad, indices)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_apply_rms_prop():
    """
    Feature: test SparseApplyRMSProp in cpu
    Description: docs params, attr and input
    Expectation: the results and expects are within 1e-6
    """
    rho = 0.2
    momentum = 0.01
    epsilon = 1e-6
    net = SparseApplyRMSPropNet(rho, momentum, epsilon)
    learning_rate = 0.01
    tol = 1e-6
    grad = np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    net.var = Parameter(Tensor(np.array([[0.6, 0.3], [0.1, 0.5]]).astype(np.float32)), name="var")
    net.ms = Parameter(Tensor(np.array([[0.2, 0.4], [0.1, 0.3]]).astype(np.float32)), name="ms")
    net.mom = Parameter(Tensor(np.array([[0.3, 0.1], [0.3, 0.6]]).astype(np.float32)), name="mom")
    output_var, output_ms, output_mom = net(learning_rate, Tensor(grad), Tensor(indices))
    expect_var = np.array([[0.5880358, 0.28881112], [0.09102397, 0.48342228]])
    expect_ms = np.array([[0.112, 0.472], [0.028, 0.572]])
    expect_mom = np.array([[0.01196417, 0.01118888], [0.00897604, 0.01657771]])
    assert (abs(output_var.asnumpy() - expect_var) <= tol).all()
    assert (abs(output_ms.asnumpy() - expect_ms) <= tol).all()
    assert (abs(output_mom.asnumpy() - expect_mom) <= tol).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_apply_rms_prop_fp32():
    """
    Feature: test SparseApplyRMSProp in cpu
    Description: normal params, attr and input in float32
    Expectation: the results and expects are within 1e-6
    """
    var = Tensor(
        np.array(
            [
                [
                    [[1.7584051, 7.845357, 9.487755, 11.609518], [6.3358746, 9.710918, 10.127965, 10.117655]],
                    [[12.163624, 5.494794, 3.8711822, 1.3894155], [8.985711, 0.6518214, 7.3151374, 16.33593]],
                    [[8.341027, 5.162506, 8.352797, 5.554555], [4.9117146, 4.477907, 13.811077, 0.54865116]],
                ],
                [
                    [[11.817743, 14.965637, 8.13786, 12.019079], [13.102469, 15.835658, 13.591752, 9.972791]],
                    [[17.454584, 11.351265, 13.24484, 3.8717928], [17.244823, 12.653173, 19.387028, 5.45228]],
                    [[18.595354, 0.32980376, 12.503356, 5.3955374], [0.47630417, 12.696551, 6.7440767, 12.151557]],
                ],
            ]
        )
    ).astype(np.float32)
    ms = Tensor(
        np.array(
            [
                [
                    [[13.247066, 3.0132513, 15.529863, 7.0405197], [15.222864, 17.862719, 14.253433, 8.52769]],
                    [[4.603761, 7.4978523, 15.64114, 3.4454918], [8.88428, 14.043913, 2.6531525, 1.7218554]],
                    [[6.9842176, 4.660216, 12.589785, 11.106893], [17.857334, 1.9999982, 2.2025642, 13.055216]],
                ],
                [
                    [[8.858172, 18.533686, 5.48135, 16.584848], [3.5365322, 2.140122, 11.01436, 1.4174879]],
                    [[18.309923, 12.984872, 16.118517, 2.7294059], [12.451426, 5.4134645, 16.591896, 4.5551147]],
                    [[5.5329094, 8.667258, 12.109718, 6.447345], [12.299871, 10.31546, 16.994408, 18.751486]],
                ],
            ]
        )
    ).astype(np.float32)
    mom = Tensor(
        np.array(
            [
                [
                    [[1.8185945, 9.377954, 0.10671406, 19.155134], [10.460225, 15.26945, 18.154474, 3.1047785]],
                    [[14.950758, 2.8664052, 9.1753845, 13.3002205], [5.3172884, 4.909375, 5.1808786, 16.881796]],
                    [[11.970335, 3.5992355, 8.939086, 10.23226], [2.2149224, 11.196065, 5.0415382, 13.498018]],
                ],
                [
                    [[19.054583, 8.202999, 5.3966255, 9.038197], [13.197036, 19.272615, 15.766206, 8.0324135]],
                    [[12.263951, 14.052368, 14.865421, 14.657042], [13.552727, 0.70198125, 2.8945522, 7.790198]],
                    [[2.3330674, 0.64346105, 19.878948, 14.215902], [18.90649, 4.7782664, 6.36722, 18.578365]],
                ],
            ]
        )
    ).astype(np.float32)
    rho = 0.2
    momentum = 0.01
    epsilon = 1e-6
    net = SparseApplyRMSPropNet(rho, momentum, epsilon, True)
    net.var = Parameter(var, name="var")
    net.ms = Parameter(ms, name="ms")
    net.mom = Parameter(mom, name="mom")
    learning_rate = 0.01
    tol = 1e-6
    grad = np.array(
        [
            [
                [[4.425984, 17.72997, 3.6272728, 14.553083], [7.809875, 1.0404425, 0.4167797, 1.4313234]],
                [[15.876797, 19.840714, 0.19511667, 8.967148], [5.1575384, 9.222021, 6.7389107, 13.391502]],
                [[3.3068883, 18.009441, 3.2276564, 8.246849], [12.699854, 18.070751, 7.0316415, 18.188854]],
            ],
            [
                [[15.942688, 10.274351, 10.572657, 6.9661407], [13.754183, 16.018494, 6.9371862, 2.9460514]],
                [[16.671234, 17.091852, 7.828639, 4.098937], [8.028752, 9.3316345, 15.868357, 1.5713477]],
                [[10.281095, 6.8612375, 0.5492036, 10.575689], [11.136571, 6.750351, 10.062054, 14.244425]],
            ],
        ]
    ).astype(np.float32)
    indices = np.array([0, 1], dtype=np.int64)
    output_var, output_ms, output_mom = net(learning_rate, Tensor(grad), Tensor(indices))
    expect_var = np.array(
        [
            [
                [[1.7298788, 7.7404103, 9.476863, 11.406833], [6.220425, 9.553286, 9.94401, 10.07878]],
                [[12.002961, 5.454976, 3.7783306, 1.2452924], [8.921798, 0.5917712, 7.252229, 16.155945]],
                [[8.210941, 5.1153536, 8.253608, 5.4412737], [4.8785367, 4.354775, 13.749543, 0.4025454]],
            ],
            [
                [[11.616066, 14.872664, 8.072782, 11.917966], [12.959345, 15.631763, 13.423217, 9.881508]],
                [[17.320856, 11.199622, 13.085356, 3.7142625], [17.098377, 12.635059, 19.346992, 5.365129]],
                [[18.560915, 0.31243756, 12.301201, 5.2422776], [0.276195, 12.637892, 6.6694517, 11.95472]],
            ],
        ]
    ).astype(np.float32)
    expect_ms = np.array(
        [
            [
                [[18.32088, 252.08414, 13.63166, 170.84189], [51.83989, 4.4385605, 2.989651, 3.3444874]],
                [[202.57889, 316.4227, 3.1586843, 65.01689], [23.057018, 70.84532, 36.860966, 143.81024]],
                [[10.145252, 260.40402, 10.852169, 56.629795], [132.60051, 261.64166, 39.9957, 267.2786]],
            ],
            [
                [[205.10707, 88.15657, 90.521126, 42.138668], [152.04935, 205.70174, 40.70252, 7.2268724]],
                [[226.00603, 236.3021, 52.253777, 13.986909], [54.058975, 70.746216, 204.76218, 2.88633]],
                [[85.667305, 39.39472, 2.6632433, 90.76563], [101.67854, 38.516884, 84.39482, 166.07321]],
            ],
        ]
    ).astype(np.float32)
    expect_mom = np.array(
        [
            [
                [[0.02852633, 0.1049465, 0.01089154, 0.2026855], [0.11544931, 0.15763302, 0.18395518, 0.03887438]],
                [[0.16066249, 0.03981787, 0.09285168, 0.14412314], [0.06391378, 0.06005022, 0.06290836, 0.1799849]],
                [[0.13008551, 0.04715267, 0.09918867, 0.11328146], [0.03317797, 0.12313244, 0.06153398, 0.14610578]],
            ],
            [
                [[0.20167777, 0.09297276, 0.06507868, 0.10111325], [0.14312467, 0.20389485, 0.16853563, 0.09128299]],
                [[0.13372889, 0.15164241, 0.15948418, 0.15753041], [0.14644705, 0.01811427, 0.0400349, 0.08715107]],
                [[0.03443857, 0.0173662, 0.20215482, 0.15325965], [0.20010917, 0.05865945, 0.07462509, 0.19683704]],
            ],
        ]
    ).astype(np.float32)
    assert (abs(output_var.asnumpy() - expect_var) <= tol).all()
    assert (abs(output_ms.asnumpy() - expect_ms) <= tol).all()
    assert (abs(output_mom.asnumpy() - expect_mom) <= tol).all()
    assert (abs(net.var.asnumpy() - expect_var) <= tol).all()
    assert (abs(net.ms.asnumpy() - expect_ms) <= tol).all()
    assert (abs(net.mom.asnumpy() - expect_mom) <= tol).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_apply_rms_prop_update_fp16():
    """
    Feature: test SparseApplyRMSProp in cpu
    Description: random params, attr and input in float16. Update net's parameters
    Expectation: the results, parameters and expects are within 1e-3
    """
    var = np.array([[[0.2048, 2.107], [3.395, 3.107]], [[1.971, 3.18], [2.648, 1.034]]])
    ms = np.array([[[4.93, 3.984], [4.25, 3.662]], [[0.6567, 4.86], [3.867, 2.898]]])
    mom = np.array([[[1.537, 1.1], [4.668, 4.03]], [[0.5044, 1.44], [3.336, 3.855]]])
    rho = 0.2
    momentum = 0.01
    epsilon = 1e-6
    tol = 1e-3
    net = SparseApplyRMSPropNet(rho, momentum, epsilon, True)
    net.var = Parameter(Tensor(var, dtype=mindspore.float16), name="var")
    net.ms = Parameter(Tensor(ms, dtype=mindspore.float16), name="ms")
    net.mom = Parameter(Tensor(mom, dtype=mindspore.float16), name="mom")
    learning_rate = Tensor(0.01, dtype=mindspore.float16)
    grad = np.array([[[4.105, 1.056], [4.773, 1.278]], [[0.5186, 1.605], [2.549, 1.029]]]).astype(np.float16)
    indices = np.array([0, 1], dtype=np.int32)
    output_var, output_ms, output_mom = net(learning_rate, Tensor(grad, dtype=mindspore.float16), Tensor(indices))
    expect_var = np.array(
        [[[0.1787, 2.08787379], [3.336, 3.05774736]], [[1.95714428, 3.15638097], [2.605, 0.98683219]]]
    ).astype(np.float16)
    expect_ms = np.array(
        [[[14.46989893, 1.68834129], [19.07856445, 2.03968226]], [[0.34645917, 3.03402393], [5.97061985, 1.42716165]]]
    ).astype(np.float16)
    expect_mom = np.array(
        [[[0.026165, 0.01912621], [0.05761078, 0.04925264]], [[0.01385572, 0.02361903], [0.04379335, 0.04716781]]]
    ).astype(np.float16)
    assert (abs(output_ms.asnumpy() - expect_ms) <= tol).all()
    assert (abs(output_var.asnumpy() - expect_var) <= tol).all()
    assert (abs(output_mom.asnumpy() - expect_mom) <= tol).all()
    assert (abs(net.var.asnumpy() - expect_var) <= tol).all()
    assert (abs(net.ms.asnumpy() - expect_ms) <= tol).all()
    assert (abs(net.mom.asnumpy() - expect_mom) <= tol).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_apply_rms_prop_grad0():
    """
    Feature: test SparseApplyRMSProp in cpu
    Description: input grad is zero
    Expectation: parameter ms is not updated, but var and mom are
    """
    rho = 0.2
    momentum = 0.01
    epsilon = 1e-6
    net = SparseApplyRMSPropNet(rho, momentum, epsilon)
    learning_rate = 0.01
    tol = 1e-6
    grad = np.array([[0, 0], [0, 0]]).astype(np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    var = np.array([[0.6, 0.3], [0.1, 0.5]]).astype(np.float32)
    ms = np.array([[0.2, 0.4], [0.1, 0.3]]).astype(np.float32)
    mom = np.array([[0.3, 0.1], [0.3, 0.6]]).astype(np.float32)
    net.var = Parameter(Tensor(var, dtype=mindspore.float32), name="var")
    net.ms = Parameter(Tensor(ms, dtype=mindspore.float32), name="ms")
    net.mom = Parameter(Tensor(mom, dtype=mindspore.float32), name="mom")
    output_var, output_ms, output_mom = net(learning_rate, Tensor(grad), Tensor(indices))
    expect_var = np.array([[0.597, 0.29900002], [0.097, 0.494]]).astype(np.float32)
    expect_ms = np.array([[0.2, 0.4], [0.1, 0.3]]).astype(np.float32)
    expect_mom = np.array([[0.003, 0.001], [0.003, 0.006]]).astype(np.float32)
    assert (abs(output_ms.asnumpy() - expect_ms) <= tol).all()
    assert (abs(output_var.asnumpy() - expect_var) <= tol).all()
    assert (abs(output_mom.asnumpy() - expect_mom) <= tol).all()
    assert (abs(net.var.asnumpy() - expect_var) <= tol).all()
    assert (abs(net.ms.asnumpy() - expect_ms) <= tol).all()
    assert (abs(net.mom.asnumpy() - expect_mom) <= tol).all()
