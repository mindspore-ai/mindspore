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

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_avgpool_k2s1pv():
    x = np.arange(1 * 1 * 6 * 6).reshape((1, 1, 6, 6)).astype(np.float32)
    net = nn.AvgPool2d(kernel_size=2, stride=1, pad_mode='valid')
    out = net(Tensor(x))
    print(out)
    expect_result = np.array(
        [[[[3.5, 4.5, 5.5, 6.5, 7.5],
           [9.5, 10.5, 11.5, 12.5, 13.5],
           [15.5, 16.5, 17.5, 18.5, 19.5],
           [21.5, 22.5, 23.5, 24.5, 25.5],
           [27.5, 28.5, 29.5, 30.5, 31.5]]]]
    )
    assert np.allclose(out.asnumpy(), expect_result)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_avgpool_k2s2pv():
    x = np.arange(1 * 1 * 6 * 6).reshape((1, 1, 6, 6)).astype(np.float32)
    net = nn.AvgPool2d(kernel_size=2, stride=2, pad_mode='valid')
    out = net(Tensor(x))
    print(out)
    expect_result = np.array(
        [[[[3.5, 5.5, 7.5],
           [15.5, 17.5, 19.5],
           [27.5, 29.5, 31.5]]]]
    )
    assert np.allclose(out.asnumpy(), expect_result)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_avgpool_k3s2pv():
    x = np.arange(1 * 1 * 6 * 6).reshape((1, 1, 6, 6)).astype(np.float32)
    net = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='valid')
    out = net(Tensor(x))
    print(out)
    expect_result = np.array(
        [[[[7., 9.],
           [19., 21.]]]]
    )
    assert np.allclose(out.asnumpy(), expect_result)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_avgpool_k3s2ps():
    x = np.arange(1 * 1 * 6 * 6).reshape((1, 1, 6, 6)).astype(np.float32)
    net = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='same')
    out = net(Tensor(x))
    print(out)
    expect_result = np.array(
        [[[[7., 9., 10.5],
           [19., 21., 22.5],
           [28., 30., 31.5]]]]
    )
    assert np.allclose(out.asnumpy(), expect_result)


if __name__ == '__main__':
    test_avgpool_k2s1pv()
    test_avgpool_k2s2pv()
    test_avgpool_k3s2pv()
    test_avgpool_k3s2ps()
