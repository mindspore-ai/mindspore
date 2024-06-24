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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def smoothl1loss(beta, reduction):
    np.random.seed(42)
    prediction = np.random.randn(20).astype(np.float32)
    target = np.random.randn(20).astype(np.float32)

    net = nn.SmoothL1Loss(beta, reduction)
    return net(Tensor(prediction), Tensor(target))


def verify_forward(reduction, loss, expect):
    if reduction == 'none':
        np.testing.assert_array_almost_equal(loss, expect)
    elif reduction == "sum":
        expect_sum = np.sum(expect)
        np.testing.assert_array_almost_equal(loss, expect_sum, decimal=5)
    elif reduction == "mean":
        expect_mean = np.mean(expect)
        np.testing.assert_array_almost_equal(loss, expect_mean)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("reduction", ['none', 'mean', 'sum'])
def test_smoothl1loss(reduction):
    """
    Feature: SmoothL1Loss cpu kernel.
    Description: test the rightness of SmoothL1Loss cpu kernel.
    Expectation: the output is same as expect.
    """

    beta = 1.0
    loss = smoothl1loss(beta, reduction)
    expect = np.array([0.46941718, 0.00382918, 0.16829303, 2.447778, 0.04812113, 0.05953304,
                       2.2302065, 0.07672881, 0.00860204, 0.34798968, 0.00956192, 1.818008,
                       0.03262977, 0.36599946, 2.047463, 0.2168481, 0.7216947, 1.7739174,
                       0.08826803, 1.109165])

    verify_forward(reduction, loss.asnumpy(), expect)

    beta = 1 / 9
    loss = smoothl1loss(beta, reduction)
    expect = np.array([0.9133791, 0.03446258, 0.5246048, 2.8922224, 0.2546738, 0.289504,
                       2.674651, 0.33618113, 0.07560876, 0.7786982, 0.08273339, 2.2624524,
                       0.19990394, 0.8000138, 2.4919074, 0.6030006, 1.1661391, 2.2183619,
                       0.3646064, 1.5536094])

    verify_forward(reduction, loss.asnumpy(), expect)
