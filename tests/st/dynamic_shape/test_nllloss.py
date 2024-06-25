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

import numpy as np
import pytest
from tests.st.utils import test_utils
from mindspore import ops
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def nllloss_forward_func(logits, labels, weight):
    return ops.NLLLoss(reduction='none', ignore_index=-100)(logits, labels, weight)


@test_utils.run_with_cell
def nllloss_backward_func(logits, labels, weight):
    return ops.grad(nllloss_forward_func, (0,))(logits, labels, weight)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@pytest.mark.parametrize('data_type', [np.float32])
@test_utils.run_test_with_On
def test_nllloss_forward(mode, data_type):
    """
    Feature: Ops.
    Description: test op nllloss.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    logits = ms.Tensor(np.array([[-1.3739, -2.2700, -3.2333, -2.4589, -0.6566],
                                 [-1.2156, -2.6026, -1.2200, -1.8731, -1.7119],
                                 [-0.7130, -3.3672, -1.5368, -1.8289, -2.3058]]).astype(data_type))
    labels = ms.Tensor(np.array([1, 0, 4]).astype(np.int32))
    weight = ms.Tensor(np.array([0.2, 0.3, 0.1, 0.15, 0.25]).astype(data_type))
    actual_output = nllloss_forward_func(logits, labels, weight)
    expect_loss = np.array([0.681, 0.24312, 0.57645]).astype(data_type)
    expect_total_weight = np.array(0.75).astype(data_type)
    assert np.allclose(actual_output[0].asnumpy(), expect_loss)
    assert np.allclose(actual_output[1].asnumpy(), expect_total_weight)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@pytest.mark.parametrize('data_type', [np.float32])
@test_utils.run_test_with_On
def test_nllloss_forward_ascend(mode, data_type):
    """
    Feature: Ops.
    Description: test op nllloss.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    logits = ms.Tensor(np.array([[-1.3739, -2.2700, -3.2333, -2.4589, -0.6566],
                                 [-1.2156, -2.6026, -1.2200, -1.8731, -1.7119],
                                 [-0.7130, -3.3672, -1.5368, -1.8289, -2.3058]]).astype(data_type))
    labels = ms.Tensor(np.array([1, 0, 4]).astype(np.int32))
    weight = ms.Tensor(np.array([0.2, 0.3, 0.1, 0.15, 0.25]).astype(data_type))
    actual_output = nllloss_forward_func(logits, labels, weight)
    expect_loss = np.array([0.681, 0.24312, 0.57645]).astype(data_type)
    expect_total_weight = np.array(0).astype(data_type)
    assert np.allclose(actual_output[0].asnumpy(), expect_loss)
    assert np.allclose(actual_output[1].asnumpy(), expect_total_weight)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@pytest.mark.parametrize('data_type', [np.float32])
@test_utils.run_test_with_On
def test_nllloss_backward(mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op nllloss.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    logits = ms.Tensor(np.array([[-1.3739, -2.2700, -3.2333, -2.4589, -0.6566],
                                 [-1.2156, -2.6026, -1.2200, -1.8731, -1.7119],
                                 [-0.7130, -3.3672, -1.5368, -1.8289, -2.3058]]).astype(data_type))
    labels = ms.Tensor(np.array([1, 0, 4]).astype(np.int32))
    weight = ms.Tensor(np.array([0.2, 0.3, 0.1, 0.15, 0.25]).astype(data_type))
    actual_grad = nllloss_backward_func(logits, labels, weight)
    except_grad = np.array([[0.0, -0.3, 0.0, 0.0, 0.0],
                            [-0.2, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, -0.25]]).astype(data_type)
    assert np.allclose(actual_grad.asnumpy(), except_grad)
