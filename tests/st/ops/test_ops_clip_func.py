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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops


class NetWorkClipByValue(nn.Cell):
    def construct(self, x, min_value, max_value):
        return ops.clip_by_value(x, min_value, max_value)


class NetWorkClipByNorm(nn.Cell):
    def construct(self, x):
        return ops.clip_by_norm(x, max_norm=1)


class NetWorkClamp(nn.Cell):
    def construct(self, x, min_value, max_value):
        return ops.clamp(x, min_value, max_value)


class NetWorkClip(nn.Cell):
    def construct(self, x, min_value, max_value):
        return ops.clip(x, min_value, max_value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clip_by_value_tensor(mode):
    """
    Feature: ops.clip_by_value
    Description: Verify the result of clip_by_value
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([-0.5962, 0.4985, 0.2349, -0.4396, 0.4525]), ms.float32)
    net = NetWorkClipByValue()
    output = net(x, -0.3, 0.4)
    expect_output = [-0.3, 0.4, 0.2349, -0.3, 0.4]
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_clip_by_value_list_tensor(mode):
    """
    Feature: ops.clip_by_value
    Description: Verify the result of clip_by_value
    Expectation: success
    """
    ms.set_context(mode=mode)
    x1 = Tensor(np.array([-0.5962, 0.4985, 0.2349, -0.4396, 0.4525]), ms.float32)
    x2 = Tensor(np.array([0.6035, 0.6959, 0.0150, -0.5766, 0.5432]), ms.float32)
    x3 = Tensor(np.array([0.7549, 0.1056, 0.3312, -0.4060, 0.9821]), ms.float32)
    net = NetWorkClipByValue()
    output = net([x1, x2, x3], -0.3, 0.4)
    expect_output = [[-0.3, 0.4, 0.2349, -0.3, 0.4],
                     [0.4, 0.4, 0.0150, -0.3, 0.4],
                     [0.4, 0.1056, 0.3312, -0.3, 0.4]
                     ]
    assert np.allclose(output[0].asnumpy(), expect_output[0])
    assert np.allclose(output[1].asnumpy(), expect_output[1])
    assert np.allclose(output[2].asnumpy(), expect_output[2])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clamp(mode):
    """
    Feature: ops.clamp
    Description: Verify the result of clamp
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([-0.5962, 0.4985, 0.2349, -0.4396, 0.4525]), ms.float32)
    net = NetWorkClamp()
    output_case_1 = net(x, -0.3, 0.4)
    expect_output_case_1 = [-0.3, 0.4, 0.2349, -0.3, 0.4]
    output_case_2 = net(x, 0.4, -0.3)
    expect_output_case_2 = [-0.3, -0.3, -0.3, -0.3, -0.3]
    assert np.allclose(output_case_1.asnumpy(), expect_output_case_1)
    assert np.allclose(output_case_2.asnumpy(), expect_output_case_2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clip(mode):
    """
    Feature: ops.clip
    Description: Verify the result of clip
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([-0.5962, 0.4985, 0.2349, -0.4396, 0.4525]), ms.float32)
    net = NetWorkClip()
    output_case_1 = net(x, -0.3, 0.4)
    expect_output_case_1 = [-0.3, 0.4, 0.2349, -0.3, 0.4]
    output_case_2 = net(x, 0.4, -0.3)
    expect_output_case_2 = [-0.3, -0.3, -0.3, -0.3, -0.3]
    assert np.allclose(output_case_1.asnumpy(), expect_output_case_1)
    assert np.allclose(output_case_2.asnumpy(), expect_output_case_2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clip_by_norm(mode):
    """
    Feature: ops.clip_by_norm
    Description: Verify the result of clip_by_norm
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[0.8748, 0.1425, 0.0076],
                   [0.7721, 0.4084, 0.0552],
                   [4.6376, 0.2914, 2.1120]])
    net = NetWorkClipByNorm()
    out = net(x)
    expect_out = np.array([[0.16650201, 0.02712224, 0.00144652],
                           [0.14695495, 0.07773139, 0.0105063],
                           [0.8826814, 0.0554626, 0.40198016]])
    assert np.allclose(out.asnumpy(), expect_out)
