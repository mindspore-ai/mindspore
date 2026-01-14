# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import mint, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def kldivloss_forward_func(input_x, target, reduction='mean', log_target=False):
    net = mint.nn.KLDivLoss(reduction=reduction, log_target=log_target)
    return net(input_x, target)

@jit(backend="ms_backend")
def kldivloss_backward_func(input_x, target, reduction='mean', log_target=False):
    net = mint.nn.KLDivLoss(reduction=reduction, log_target=log_target)
    return ms.grad(net, (0,))(input_x, target)

def mint_nn_kldivloss_binary_compare(input_binary_data, output_binary_data, reduction='mean',
                                     log_target=False, compare_grad=True):
    input_x = ms.Tensor(input_binary_data[0])
    target = ms.Tensor(input_binary_data[1])

    output = kldivloss_forward_func(input_x, target, reduction, log_target)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    if compare_grad:
        grad = kldivloss_backward_func(input_x, target, reduction, log_target)
        assert np.allclose(grad.asnumpy(), output_binary_data[1], 1e-04, 1e-04)

@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((2, 3), np.float32),
            ((2, 3), np.float32),
        ],
        output_info=[
            ((), np.float32),
            ((2, 3), np.float32),
        ],
        extra_info='SD5B'
    )
)
def mint_nn_kldivloss_binary_case1(input_binary_data=None, output_binary_data=None):
    mint_nn_kldivloss_binary_compare(input_binary_data, output_binary_data, 'mean', True)

@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((2, 3, 4), np.float32),
            ((2, 3, 4), np.float32),
        ],
        output_info=[
            ((), np.float32),
            ((2, 3, 4), np.float32),
        ],
        extra_info='SD5B'
    )
)
def mint_nn_kldivloss_binary_case2(input_binary_data=None, output_binary_data=None):
    mint_nn_kldivloss_binary_compare(input_binary_data, output_binary_data, 'sum', False)

@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((2, 3, 4, 5), np.float32),
            ((2, 3, 4, 5), np.float32),
        ],
        output_info=[
            ((), np.float32),
            ((2, 3, 4, 5), np.float32),
        ],
        extra_info='SD5B'
    )
)
def mint_nn_kldivloss_binary_case3(input_binary_data=None, output_binary_data=None):
    mint_nn_kldivloss_binary_compare(input_binary_data, output_binary_data, 'batchmean', False)

@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((2, 3, 4, 5, 6), np.float32),
            ((2, 3, 4, 5, 6), np.float32),
        ],
        output_info=[
            ((2, 3, 4, 5, 6), np.float32),
            ((2, 3, 4, 5, 6), np.float32),
        ],
        extra_info='SD5B'
    )
)
def mint_nn_kldivloss_binary_case4(input_binary_data=None, output_binary_data=None):
    mint_nn_kldivloss_binary_compare(input_binary_data, output_binary_data, 'none', True)

@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((2, 3, 4), np.float32),
            ((2, 1, 4), np.float32),
        ],
        output_info=[
            ((), np.float32),
            ((2, 3, 4), np.float32),
        ],
        extra_info='SD5B'
    )
)
def mint_nn_kldivloss_binary_case5(input_binary_data=None, output_binary_data=None):
    mint_nn_kldivloss_binary_compare(input_binary_data, output_binary_data, 'mean', False)

@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((2, 3, 4), np.float32),
            ((3, 4), np.float32),
        ],
        output_info=[
            ((2, 3, 4), np.float32),
            ((2, 3, 4), np.float32),
        ],
        extra_info='SD5B'
    )
)
def mint_nn_kldivloss_binary_case6(input_binary_data=None, output_binary_data=None):
    mint_nn_kldivloss_binary_compare(input_binary_data, output_binary_data, 'none', True)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_mint_nn_kldivloss_normal(mode):
    """
    Feature: mint.nn.KLDivLoss.
    Description: verify the result of KLDivLoss with binary cases.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    mint_nn_kldivloss_binary_case1()
    mint_nn_kldivloss_binary_case2()
    mint_nn_kldivloss_binary_case3()
    mint_nn_kldivloss_binary_case4()

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_mint_nn_kldivloss_broadcast(mode):
    """
    Feature: mint.nn.KLDivLoss.
    Description: verify the result of KLDivLoss with broadcast cases.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    mint_nn_kldivloss_binary_case5()
    mint_nn_kldivloss_binary_case6()

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none', 'batchmean'])
def test_mint_nn_kldivloss_dynamic(reduction):
    """
    Feature: mint.nn.KLDivLoss.
    Description: test mint.nn.KLDivLoss with dynamic shape and dynamic rank.
    Expectation: expect correct result.
    """
    input1 = ms.Tensor(np.random.rand(3, 4, 5).astype(np.float32))
    target1 = ms.Tensor(np.random.rand(3, 4, 5).astype(np.float32))

    input2 = ms.Tensor(np.random.rand(5, 3).astype(np.float32))
    target2 = ms.Tensor(np.random.rand(5, 3).astype(np.float32))

    # TEST_OP Todo: ScalarTensor failed with reduction='none'
    net = mint.nn.KLDivLoss(reduction=reduction)
    TEST_OP(
        net,
        [[input1, target1], [input2, target2]],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['ScalarTensor'],
        case_config={'disable_input_check': True,
                     'all_dim_zero': True}
    )
