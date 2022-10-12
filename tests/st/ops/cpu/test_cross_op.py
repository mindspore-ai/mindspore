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
import pytest
import numpy as np
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


def test_cross_functional_api():
    """
    Feature: test cross functional API.
    Description: test case for cross functional API.
    Expectation: the result match with expected result.
    """
    a = Tensor([[-0.3956, 1.1455, 1.6895],
                [-0.5849, 1.3672, 0.3599],
                [-1.1626, 0.7180, -0.0521],
                [-0.1339, 0.9902, -2.0225]], mstype.float32)
    b = Tensor([[-0.0257, -1.4725, -1.2251],
                [-1.1479, -0.7005, -1.9757],
                [-1.3904, 0.3726, -1.1836],
                [-0.9688, -0.7153, 0.2159]], mstype.float32)
    output = F.cross(a, b, dim=1)
    expected = np.array([[1.084437, -0.52807, 0.61196],
                         [-2.449067, -1.568716, 1.979131],
                         [-0.830412, -1.303614, 0.565122],
                         [-1.23291, 1.988307, 1.055084]], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def test_cross_tensor_api():
    """
    Feature: test cross tensor API.
    Description: test case for cross tensor API.
    Expectation: the result match with expected result.
    """
    a = Tensor([[-0.3956, 1.1455, 1.6895],
                [-0.5849, 1.3672, 0.3599],
                [-1.1626, 0.7180, -0.0521],
                [-0.1339, 0.9902, -2.0225]], mstype.float32)
    b = Tensor([[-0.0257, -1.4725, -1.2251],
                [-1.1479, -0.7005, -1.9757],
                [-1.3904, 0.3726, -1.1836],
                [-0.9688, -0.7153, 0.2159]], mstype.float32)
    output = a.cross(b, dim=1)
    expected = np.array([[1.084437, -0.52807, 0.61196],
                         [-2.449067, -1.568716, 1.979131],
                         [-0.830412, -1.303614, 0.565122],
                         [-1.23291, 1.988307, 1.055084]], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cross_functional_tensor_modes():
    """
    Feature: test cross functional and tensor APIs in PyNative and Graph modes.
    Description: test case for cross functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_cross_functional_api()
    test_cross_tensor_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_cross_functional_api()
    test_cross_tensor_api()


if __name__ == '__main__':
    test_cross_functional_tensor_modes()
