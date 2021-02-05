# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class GeluNet(nn.Cell):
    def __init__(self):
        super(GeluNet, self).__init__()
        self.gelu = P.GeLU()

    def construct(self, x):
        return self.gelu(x)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        gout = self.grad(self.network)(input_data, sens)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gelugrad():
    x_ms = Tensor(np.array([0.58401114, 0.68800163, 0.9760397, 0.14702141, 0.46563736, 0.9607501,
                            0.14567593, 0.12261796, 0.37054458, 0.46421242]).astype(np.float32))
    dy_ms = Tensor(np.array([0.5559598, 0.96994054, 0.24770357, 0.34646875, 0.2984393, 0.03287048,
                             0.55681044, 0.966908, 0.06015943, 0.6099489]).astype(np.float32))

    net = GeluNet()
    grad = Grad(net)

    output = grad(x_ms, dy_ms)
    expect = [0.50963277, 0.9414753, 0.2667653, 0.21358444, 0.25243032, 0.0352667,
              0.34266686, 0.57757664, 0.04707306, 0.51536125]
    assert np.allclose(output[0].asnumpy(), expect)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gelugrad_fp16():
    np.random.seed(42)
    x_np = np.random.randn(5, 3, 6).astype(np.float16)
    dy_np = np.random.randn(5, 3, 6).astype(np.float16)
    net = GeluNet()
    grad = Grad(net)
    output = grad(Tensor(x_np), Tensor(dy_np))
    expect = [[[8.4045e-02, 3.7817e-01, -6.6748e-01, -3.6914e-01, -1.2415e-01, -4.6362e-01],
               [3.3301e-01, 2.6270e-01, 7.7534e-04, -2.0947e-01, -2.2021e-01, -6.4880e-02],
               [-2.3633e-01, 7.6538e-02, 1.8280e-02, 3.8635e-02, -1.6235e-01, 1.2964e-01]],

              [[-1.4801e-02, 9.6130e-03, -2.1660e+00, -8.5602e-03, 3.3356e-02, -3.1885e-01],
               [-2.0355e-02, 1.7737e-01, 3.8719e-03, -9.1895e-01, 8.4717e-02, 2.0593e-01],
               [5.8350e-02, -1.0020e+00, 6.8652e-01, 1.3428e-01, 6.0352e-01, -2.6270e-01]],

              [[-6.5820e-01, 5.1147e-02, -1.2650e-02, -3.2983e-01, -1.5410e+00, 4.3518e-02],
               [-4.3359e-01, 1.2659e-01, 1.1792e-01, 2.2705e-02, -1.2329e-01, -3.5278e-01],
               [6.2109e-01, 1.3611e-01, 1.7041e-01, 2.7124e-01, -5.5908e-02, 1.7212e-01]],

              [[2.8320e-01, 8.3252e-01, 4.2480e-02, -3.4473e-01, 3.9429e-01, 3.1958e-01],
               [3.6499e-02, 1.2250e-01, 7.1350e-02, -2.7267e-02, 3.0029e-01, -8.0566e-01],
               [8.2617e-01, 5.1367e-01, -9.2480e-01, 3.3203e-02, -7.5684e-01, 8.8623e-01]],

              [[5.4590e-01, -9.2383e-01, -2.8107e-02, 4.2432e-01, 4.6826e-01, 5.0879e-01],
               [-1.4062e-01, 6.6284e-02, -2.9126e-01, -6.3086e-01, -8.6975e-02, 4.1504e-02],
               [-6.3171e-03, 1.0852e-01, 1.3779e-02, 1.0947e+00, -3.0334e-02, 2.3828e+00]]]
    assert np.allclose(output[0].asnumpy(), expect, rtol=1e-2)
