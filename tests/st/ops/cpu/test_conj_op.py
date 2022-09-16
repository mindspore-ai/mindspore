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


def test_conj_functional_api():
    """
    Feature: test conj functional API.
    Description: test case for conj functional API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array((1.3 + 0.4j)), mstype.complex64)
    output = F.conj(x)
    expected = np.array((1.3 - 0.4j), np.complex64)
    np.testing.assert_array_equal(output.asnumpy(), expected)


def test_conj_tensor_api():
    """
    Feature: test conj tensor API.
    Description: test case for conj tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array((1.3 + 0.4j)), mstype.complex64)
    output = x.conj()
    expected = np.array((1.3 - 0.4j), np.complex64)
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_conj_functional_tensor_modes():
    """
    Feature: test conj functional and tensor APIs in PyNative and Graph modes.
    Description: test case for conj functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_conj_functional_api()
    test_conj_tensor_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_conj_functional_api()
    test_conj_tensor_api()


if __name__ == '__main__':
    test_conj_functional_tensor_modes()
