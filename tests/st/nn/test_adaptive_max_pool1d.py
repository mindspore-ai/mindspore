# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore import Tensor


class AdaptiveMaxPool1dNet(nn.Cell):
    """AdaptiveMaxPool1d."""

    def __init__(self, output_size, return_indices):
        super(AdaptiveMaxPool1dNet, self).__init__()
        self.adaptive_max_pool_1d = nn.AdaptiveMaxPool1d(output_size, return_indices)

    def construct(self, x):
        return self.adaptive_max_pool_1d(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nn_adaptivemaxpool1d_2d(mode):
    """
    Feature: nn.AdaptiveMaxPool1d
    Description: Verify the result of AdaptiveMaxPool1d
    Expectation: success
    """
    ms.set_context(mode=mode)
    a = np.arange(12).reshape(3, 4).astype(np.float32)
    x = Tensor(a)
    except_out_val = np.array([[1., 2., 3.],
                               [5., 6., 7.],
                               [9., 10., 11.]], dtype=np.float32)
    except_out_indices = np.array([[1, 2, 3],
                                   [1, 2, 3],
                                   [1, 2, 3]])
    if ms.get_context("device_target") == "Ascend":
        return_indices = False
    else:
        return_indices = True
    net = AdaptiveMaxPool1dNet(3, return_indices)
    out = net(x)
    if return_indices:
        assert np.allclose(out[0].asnumpy(), except_out_val)
        assert np.array_equal(out[1].asnumpy(), except_out_indices)
    else:
        assert np.allclose(out.asnumpy(), except_out_val)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nn_adaptivemaxpool1d_3d(mode):
    """
    Feature: nn.AdaptiveMaxPool1d
    Description: Verify the result of AdaptiveMaxPool1d
    Expectation: success
    """
    ms.set_context(mode=mode)
    a = np.arange(16).reshape(2, 2, 4).astype(np.float32)
    x = Tensor(a)
    except_out_val = np.array([[[1., 2., 3.],
                                [5., 6., 7.]],
                               [[9., 10., 11.],
                                [13., 14., 15.]]], dtype=np.float32)
    except_out_indices = np.array([[[1, 2, 3],
                                    [1, 2, 3]],
                                   [[1, 2, 3],
                                    [1, 2, 3]]])
    if ms.get_context("device_target") == "Ascend":
        return_indices = False
    else:
        return_indices = True
    net = AdaptiveMaxPool1dNet(3, return_indices)
    out = net(x)
    if return_indices:
        assert np.allclose(out[0].asnumpy(), except_out_val)
        assert np.array_equal(out[1].asnumpy(), except_out_indices)
    else:
        assert np.allclose(out.asnumpy(), except_out_val)
