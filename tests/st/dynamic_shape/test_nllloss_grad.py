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
import mindspore as ms
from mindspore import ops
from tests.st.utils import test_utils
import pytest
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def nllloss_grad_forward_func(logits, loss_grad, labels, weight, total_weight):
    return ops.auto_generate.NLLLossGrad(reduction='none', ignore_index=-100)(logits, loss_grad, labels, weight,
                                                                              total_weight)

@test_utils.run_with_cell
def nllloss_grad_vmap_func(logits, loss_grad, labels, weight, total_weight):
    return ops.vmap(nllloss_grad_forward_func, in_axes=(0, 0, 0, None, None), out_axes=0)(logits, loss_grad, labels,
                                                                                          weight, total_weight)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('data_type', [np.float32])
@test_utils.run_test_with_On
def test_nllloss_grad(mode, data_type):
    """
    Feature: DynamicShape.
    Description: Create NLLLossGrad instance with constant arguaments.
    Expectation: No exception.
    """
    ms.context.set_context(mode=mode)
    logits = ms.Tensor(np.array([[-1.3739, -2.2700, -3.2333, -2.4589, -0.6566],
                                 [-1.2156, -2.6026, -1.2200, -1.8731, -1.7119],
                                 [-0.7130, -3.3672, -1.5368, -1.8289, -2.3058]]).astype(data_type))
    labels = ms.Tensor(np.array([1, 0, 4]).astype(np.int32))
    weight = ms.Tensor(np.array([0.2, 0.3, 0.1, 0.15, 0.25]).astype(data_type))
    loss_grad = ms.Tensor(np.arange(3).astype(data_type).reshape(3))
    total_weight = ms.Tensor(np.arange(1).astype(data_type).reshape(1))
    out = nllloss_grad_forward_func(logits, loss_grad, labels, weight, total_weight)
    expect_out = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [-0.2, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, -0.5]]).astype(data_type)
    assert np.allclose(out.asnumpy(), expect_out)

def get_grad_inputs_and_output(nptype_input, nptype_weight, reduction, input_type="Tensor"):
    """Get inputs and outputs for nll loss grad."""
    x = np.array([[0.53, 0.74, -2.12], [1.29, -0.34, -1.13]]).astype(nptype_input)

    if reduction == "none":
        dloss = np.array([3.24, -2.13]).astype(nptype_input)
    else:
        dloss = np.array(1.23).astype(nptype_input)

    target = np.array([0, 1]).astype(np.int32)
    weight = np.array([0.45, -0.32, 1.21]).astype(nptype_weight)

    total_weight = np.array(0.13).astype(nptype_weight)

    inputs = (x, dloss, target, weight, total_weight)
    if input_type == "Tensor":
        inputs = (ms.Tensor(input_element) for input_element in inputs)

    if reduction == "none":
        dx_expected = np.array([[-1.45799994, 0, 0], [0, -0.681600034, 0]])
    elif reduction == "mean":
        dx_expected = np.array([[-4.25769234, 0, 0], [0, 3.02769232, 0]])
    else:
        dx_expected = np.array([[-0.553499997, 0, 0], [0, 0.393599987, 0]])

    outputs = (dx_expected,)
    return inputs, outputs


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('data_type', [np.float32])
@test_utils.run_test_with_On
def test_nllloss_grad_vmap(mode, data_type):
    """
    Feature: test NLLLossGrad vmap interface.
    Description: test the rightness of NLLLossGrad kernel.
    Expectation: the result match with numpy result
    """
    ms.context.set_context(mode=mode)
    inputs, expected_outputs = get_grad_inputs_and_output(data_type, data_type, "none", "numpy")
    x, dloss, target, weight, total_weight = inputs
    dim_size = 3
    stack_x = np.stack([x] * dim_size)
    stack_dloss = np.stack([dloss] * dim_size)
    stack_target = np.stack([target] * dim_size)

    outputs = nllloss_grad_vmap_func(ms.Tensor(stack_x), ms.Tensor(stack_dloss), ms.Tensor(stack_target),
                                     ms.Tensor(weight), ms.Tensor(total_weight))
    expect = np.stack([expected_outputs[0]] * dim_size)
    ertol_loss = 1e-06
    np.testing.assert_allclose(outputs.asnumpy(), expect, ertol_loss)
