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
from mindspore import mint
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def bceloss_forward_func(logits, labels, weight=None, reduction='mean'):
    net = mint.nn.BCELoss(weight=weight, reduction=reduction)
    return net(logits, labels)

def bceloss_backward_func(logits, labels, weight=None, reduction='mean'):
    net = mint.nn.BCELoss(weight=weight, reduction=reduction)
    return ms.grad(net, (0, 1))(logits, labels)

def mint_nn_bceloss_binary_compare(input_binary_data, output_binary_data, weight=None, reduction='mean'):
    logits = ms.Tensor(input_binary_data[0])
    labels = ms.Tensor(input_binary_data[1])

    output = bceloss_forward_func(logits, labels, weight, reduction)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    grads = bceloss_backward_func(logits, labels, weight, reduction)
    assert np.allclose(grads[0].asnumpy(), output_binary_data[1], 1e-04, 1e-04)
    assert np.allclose(grads[1].asnumpy(), output_binary_data[2], 1e-04, 1e-04)

@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((2, 3), np.float32),
            ((2, 3), np.float32),
            ((2, 3), np.float32),
        ],
        output_info=[
            ((), np.float32),
            ((2, 3), np.float32),
            ((2, 3), np.float32),
        ],
        extra_info='SD5B'
    )
)
def mint_nn_bceloss_binary_case1(input_binary_data=None, output_binary_data=None):
    mint_nn_bceloss_binary_compare(input_binary_data, output_binary_data, ms.Tensor(input_binary_data[2]), 'mean')

@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((2, 3, 4), np.float32),
            ((2, 3, 4), np.float32),
            ((2, 3, 4), np.float32),
        ],
        output_info=[
            ((), np.float32),
            ((2, 3, 4), np.float32),
            ((2, 3, 4), np.float32),
        ],
        extra_info='SD5B'
    )
)
def mint_nn_bceloss_binary_case2(input_binary_data=None, output_binary_data=None):
    mint_nn_bceloss_binary_compare(input_binary_data, output_binary_data, ms.Tensor(input_binary_data[2]), 'sum')

@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((2, 3, 4), np.float32),
            ((2, 3, 4), np.float32),
            ((2, 3, 4), np.float32),
        ],
        output_info=[
            ((2, 3, 4), np.float32),
            ((2, 3, 4), np.float32),
            ((2, 3, 4), np.float32),
        ],
        extra_info='SD5B'
    )
)
def mint_nn_bceloss_binary_case3(input_binary_data=None, output_binary_data=None):
    mint_nn_bceloss_binary_compare(input_binary_data, output_binary_data, ms.Tensor(input_binary_data[2]), 'none')

@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((2, 3, 4, 5), np.float32),
            ((2, 3, 4, 5), np.float32),
        ],
        output_info=[
            ((), np.float32),
            ((2, 3, 4, 5), np.float32),
            ((2, 3, 4, 5), np.float32),
        ],
        extra_info='SD5B'
    )
)
def mint_nn_bceloss_binary_case4(input_binary_data=None, output_binary_data=None):
    mint_nn_bceloss_binary_compare(input_binary_data, output_binary_data, None, 'mean')

@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((2, 3, 4, 6), np.float32),
            ((2, 3, 4, 6), np.float32),
        ],
        output_info=[
            ((), np.float32),
            ((2, 3, 4, 6), np.float32),
            ((2, 3, 4, 6), np.float32),
        ],
        extra_info='SD5B'
    )
)
def mint_nn_bceloss_binary_case5(input_binary_data=None, output_binary_data=None):
    mint_nn_bceloss_binary_compare(input_binary_data, output_binary_data, None, 'sum')

@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((2, 3, 4, 5, 6), np.float32),
            ((2, 3, 4, 5, 6), np.float32),
        ],
        output_info=[
            ((2, 3, 4, 5, 6), np.float32),
            ((2, 3, 4, 5, 6), np.float32),
            ((2, 3, 4, 5, 6), np.float32),
        ],
        extra_info='SD5B'
    )
)
def mint_nn_bceloss_binary_case6(input_binary_data=None, output_binary_data=None):
    mint_nn_bceloss_binary_compare(input_binary_data, output_binary_data, None, 'none')

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_mint_nn_bceloss_binary_cases(mode):
    """
    Feature: mint.nn.BCELoss.
    Description: verify the result of BCELoss with binary cases.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    mint_nn_bceloss_binary_case1()
    mint_nn_bceloss_binary_case2()
    mint_nn_bceloss_binary_case3()
    mint_nn_bceloss_binary_case4()
    mint_nn_bceloss_binary_case5()
    mint_nn_bceloss_binary_case6()

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_mint_nn_bceloss_dynamic(reduction):
    """
    Feature: mint.nn.BCELoss.
    Description: test function var_forward_func with dynamic shape and dynamic rank.
    Expectation: expect correct result.
    """
    logits1 = ms.Tensor(np.random.rand(3, 4, 5).astype(np.float32))
    labels1 = ms.Tensor(np.random.rand(3, 4, 5).astype(np.float32))

    logits2 = ms.Tensor(np.random.rand(5, 3).astype(np.float32))
    labels2 = ms.Tensor(np.random.rand(5, 3).astype(np.float32))

    net = mint.nn.BCELoss(reduction=reduction)
    TEST_OP(
        net,
        [[logits1, labels1], [logits2, labels2]],
        disable_mode=["GRAPH_MODE_GE"],
        case_config={'disable_input_check': True,
                     'all_dim_zero': True}
    )
