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
""" test graph JIT Fallback runtime feature """
import pytest
import numpy as np

import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)


class ConstNet(ms.nn.Cell):
    def np_function(self, a, b):
        return np.exp(a.asnumpy() + b.asnumpy())

    def construct(self):
        a = ms.Tensor(np.array(4), ms.int32)
        b = ms.Tensor(np.array(5), ms.int32)
        return self.np_function(a, b)


class Net(ms.nn.Cell):
    def np_function(self, a, b):
        return np.exp(a.asnumpy() + b.asnumpy())

    def np_function2(self, a, b):
        x = np.exp(a.asnumpy())
        y = np.exp(b.asnumpy())
        return np.exp(x + y)

    def construct(self, a, b):
        return self.np_function(a, b)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_np():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    a = ms.Tensor(np.array(4), ms.int32)
    b = ms.Tensor(np.array(5), ms.int32)
    output = Net()(a, b)
    print(f'output: {output}')
    const_output = ConstNet()()
    print(f'const_output: {const_output}')
    np.testing.assert_almost_equal(output, const_output, 3)


class Net1(ms.nn.Cell):
    def np_function(self, a, b):
        x = a.asnumpy()
        y = b.asnumpy()
        return np.exp(x + y)

    def construct(self, a, b):
        return self.np_function(a, b)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_np_asnumpy():
    """
    Feature: Support JIT Fallback runtime feature.
    Description: Support JIT Fallback runtime feature.
    Expectation: No exception.
    """
    a = ms.Tensor(np.array(4), ms.int32)
    b = ms.Tensor(np.array(5), ms.int32)
    output = Net1()(a, b)
    print(f'output: {output}')
    const_output = ConstNet()()
    print(f'const_output: {const_output}')
    np.testing.assert_almost_equal(output, const_output, 3)
