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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import mint
from mindspore.ops.composite import GradOperation
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


class EinsumNet(nn.Cell):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def construct(self, *operands):
        return mint.einsum(self.equation, *operands)


class EinsumGradNet(nn.Cell):
    def __init__(self, net):
        super(EinsumGradNet, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=False)
        self.net = net

    def construct(self, *operands):
        return self.grad(self.net)(*operands)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def mint_einsum_binary_case_compare(equation, input_binary_data, output_binary_data, loss=1e-4):
    inputs = [ms.Tensor(np_data) for np_data in input_binary_data]
    expect_grads = output_binary_data[1:]
    output = EinsumNet(equation)(*inputs)
    np.testing.assert_allclose(output.asnumpy(), output_binary_data[0], rtol=loss)
    grads = EinsumGradNet(EinsumNet(equation))(*inputs)
    for idx, expect_grad in enumerate(expect_grads):
        np.testing.assert_allclose(grads[idx].asnumpy(), expect_grad, rtol=loss)


@ops_binary_cases(OpsBinaryCase(input_info=[((64, 32, 16), np.float32)],
                                output_info=[((), np.float32), ((64, 32, 16), np.float32)]))
def mint_einsum_binary_case1(input_binary_data=None, output_binary_data=None):
    mint_einsum_binary_case_compare('abc -> ', input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((32, 32, 16), np.float32)],
                                output_info=[((32, 16), np.float32), ((32, 32, 16), np.float32)]))
def mint_einsum_binary_case2(input_binary_data=None, output_binary_data=None):
    mint_einsum_binary_case_compare('aab -> ab', input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((32, 16, 8), np.float32)],
                                output_info=[((8, 32, 16), np.float32), ((32, 16, 8), np.float32)]))
def mint_einsum_binary_case3(input_binary_data=None, output_binary_data=None):
    mint_einsum_binary_case_compare('abc -> cab', input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((32, 16), np.float32), ((32, 16), np.float32)],
                                output_info=[((32, 16), np.float32), ((32, 16), np.float32), ((32, 16), np.float32)]))
def mint_einsum_binary_case4(input_binary_data=None, output_binary_data=None):
    mint_einsum_binary_case_compare('ab, ab -> ab', input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((32, 16), np.float32), ((32, 16), np.float32)],
                                output_info=[((), np.float32), ((32, 16), np.float32), ((32, 16), np.float32)]))
def mint_einsum_binary_case5(input_binary_data=None, output_binary_data=None):
    mint_einsum_binary_case_compare('ab, ab -> ', input_binary_data, output_binary_data, 1e-3)


@ops_binary_cases(OpsBinaryCase(input_info=[((32, 16, 8), np.float32), ((16, 8, 12), np.float32)],
                                output_info=[((32, 12), np.float32), ((32, 16, 8), np.float32),
                                             ((16, 8, 12), np.float32)]))
def mint_einsum_binary_case6(input_binary_data=None, output_binary_data=None):
    mint_einsum_binary_case_compare('abc, bcd -> ad', input_binary_data, output_binary_data, 4e-3)


@ops_binary_cases(OpsBinaryCase(input_info=[((32, 16, 8, 12, 4, 3), np.float32), ((4, 16, 8, 12), np.float32),
                                            ((3, 32), np.float32)],
                                output_info=[((3, 32), np.float32), ((32, 16, 8, 12, 4, 3), np.float32),
                                             ((4, 16, 8, 12), np.float32), ((3, 32), np.float32)]))
def mint_einsum_binary_case7(input_binary_data=None, output_binary_data=None):
    mint_einsum_binary_case_compare('a ... dcb, c...d, ba -> ba', input_binary_data, output_binary_data, 4e-3)


@ops_binary_cases(OpsBinaryCase(input_info=[((32, 1), np.float32), ((1, 16), np.float32)],
                                output_info=[((32, 16), np.float32), ((32, 1), np.float32), ((1, 16), np.float32)]))
def mint_einsum_binary_case8(input_binary_data=None, output_binary_data=None):
    mint_einsum_binary_case_compare('ab, ab -> ab', input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((32, 1), np.float32), ((1, 16), np.float32)],
                                output_info=[((), np.float32), ((32, 1), np.float32), ((1, 16), np.float32)]))
def mint_einsum_binary_case9(input_binary_data=None, output_binary_data=None):
    mint_einsum_binary_case_compare('ab, ab -> ', input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((32, 1, 8), np.float32), ((16, 1, 12), np.float32)],
                                output_info=[((32, 12), np.float32), ((32, 1, 8), np.float32),
                                             ((16, 1, 12), np.float32)]))
def mint_einsum_binary_case10(input_binary_data=None, output_binary_data=None):
    mint_einsum_binary_case_compare('a..., ...d -> ad', input_binary_data, output_binary_data, 4e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_mint_einsum_binary_cases(mode):
    """
    Feature: standard forward, backward features.
    Description: test function einsum.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    mint_einsum_binary_case1()
    mint_einsum_binary_case2()
    mint_einsum_binary_case3()
    mint_einsum_binary_case4()
    mint_einsum_binary_case5()
    mint_einsum_binary_case8()
    mint_einsum_binary_case9()
    mint_einsum_binary_case10()


class EinsumSubListNet(nn.Cell):
    def __init__(self, in_sublist, out_sublist):
        super().__init__()
        self.in_sublist = in_sublist
        self.out_sublist = out_sublist

    def construct(self, *operands):
        sublist_inputs = []
        for idx, operand in enumerate(operands):
            sublist_inputs.append(operand)
            sublist_inputs.append(self.in_sublist[idx])
        if self.out_sublist is not None:
            sublist_inputs.append(self.out_sublist)
        return mint.einsum(*sublist_inputs)


def mint_einsum_binary_case_compare_sublist(in_sublist, out_sublist, input_binary_data, output_binary_data, loss=1e-4):
    inputs = [ms.Tensor(np_data) for np_data in input_binary_data]
    expect_grads = output_binary_data[1:]
    output = EinsumSubListNet(in_sublist, out_sublist)(*inputs)
    np.testing.assert_allclose(output.asnumpy(), output_binary_data[0], rtol=loss)
    grads = EinsumGradNet(EinsumSubListNet(in_sublist, out_sublist))(*inputs)
    for idx, expect_grad in enumerate(expect_grads):
        np.testing.assert_allclose(grads[idx].asnumpy(), expect_grad, rtol=loss)


@ops_binary_cases(OpsBinaryCase(input_info=[((16, 12, 8), np.float32)],
                                output_info=[((), np.float32), ((16, 12, 8), np.float32)]))
def mint_einsum_sublist_binary_case1(input_binary_data=None, output_binary_data=None):
    in_sublist = [[23, 24, 25]]
    out_sublist = []
    mint_einsum_binary_case_compare_sublist(in_sublist, out_sublist, input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((16, 12, 8), np.float32)],
                                output_info=[((12, 8, 16), np.float32), ((16, 12, 8), np.float32)]))
def mint_einsum_sublist_binary_case2(input_binary_data=None, output_binary_data=None):
    in_sublist = [[23, ...]]
    out_sublist = [..., 23]
    mint_einsum_binary_case_compare_sublist(in_sublist, out_sublist, input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((16, 12, 8), np.float32), ((12, 8, 15), np.float32)],
                                output_info=[((12, 8, 16, 15), np.float32), ((16, 12, 8), np.float32),
                                             ((12, 8, 15), np.float32)]))
def mint_einsum_sublist_binary_case3(input_binary_data=None, output_binary_data=None):
    in_sublist = [[23, ...], [..., 24]]
    out_sublist = None
    mint_einsum_binary_case_compare_sublist(in_sublist, out_sublist, input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((16, 16, 12, 8), np.float32), ((2, 12, 8, 15), np.float32),
                                            ((16, 15), np.float32)],
                                output_info=[((15, 16), np.float32), ((16, 16, 12, 8), np.float32),
                                             ((2, 12, 8, 15), np.float32), ((16, 15), np.float32)]))
def mint_einsum_sublist_binary_case4(input_binary_data=None, output_binary_data=None):
    in_sublist = [[11, 11, ...], [7, ..., 51], [11, 51]]
    out_sublist = [51, 11]
    mint_einsum_binary_case_compare_sublist(in_sublist, out_sublist, input_binary_data, output_binary_data, 4e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_mint_einsum_sublist_binary_cases(mode):
    """
    Feature: standard forward, backward features, use sublist mode.
    Description: test function einsum.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    mint_einsum_sublist_binary_case1()
    mint_einsum_sublist_binary_case2()
    mint_einsum_sublist_binary_case3()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=10.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_mint_einsum_bmm_binary_cases(mode):
    """
    Feature: standard forward, backward features, test bmm in einsum.
    Description: test function einsum.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    mint_einsum_binary_case6()
    mint_einsum_binary_case7()
    mint_einsum_sublist_binary_case4()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_mint_einsum_dynamic():
    """
    Feature: dynamic support.
    Description: test function einsum.
    Expectation: expect correct result.
    """
    net = EinsumNet('abc -> ')
    inputs1 = [ms.Tensor(generate_random_input((2, 3, 4), dtype=np.float32))]
    inputs2 = [ms.Tensor(generate_random_input((5, 6, 7), dtype=np.float32))]
    TEST_OP(net, [inputs1, inputs2],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True},
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'])

    net = EinsumNet('aa -> a')
    inputs1 = [ms.Tensor(generate_random_input((2, 2), dtype=np.float32))]
    inputs2 = [ms.Tensor(generate_random_input((3, 3), dtype=np.float32))]
    TEST_OP(net, [inputs1, inputs2],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True},
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'])

    net = EinsumNet('abc -> cab')
    inputs1 = [ms.Tensor(generate_random_input((2, 3, 4), dtype=np.float32))]
    inputs2 = [ms.Tensor(generate_random_input((5, 6, 7), dtype=np.float32))]
    TEST_OP(net, [inputs1, inputs2],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True},
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'])

    net = EinsumNet('ab, ab -> ab')
    inputs1 = [ms.Tensor(generate_random_input((2, 3), dtype=np.float32)),
               ms.Tensor(generate_random_input((2, 3), dtype=np.float32))]
    inputs2 = [ms.Tensor(generate_random_input((4, 5), dtype=np.float32)),
               ms.Tensor(generate_random_input((4, 5), dtype=np.float32))]
    TEST_OP(net, [inputs1, inputs2],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True},
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'])

    net = EinsumNet('ab, ab -> ')
    inputs1 = [ms.Tensor(generate_random_input((2, 3), dtype=np.float32)),
               ms.Tensor(generate_random_input((2, 3), dtype=np.float32))]
    inputs2 = [ms.Tensor(generate_random_input((4, 5), dtype=np.float32)),
               ms.Tensor(generate_random_input((4, 5), dtype=np.float32))]
    TEST_OP(net, [inputs1, inputs2],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True},
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'])

    net = EinsumNet('abc, bcd -> ad')
    inputs1 = [ms.Tensor(generate_random_input((2, 3, 4), dtype=np.float32)),
               ms.Tensor(generate_random_input((3, 4, 5), dtype=np.float32))]
    inputs2 = [ms.Tensor(generate_random_input((4, 5, 6), dtype=np.float32)),
               ms.Tensor(generate_random_input((5, 6, 7), dtype=np.float32))]
    TEST_OP(net, [inputs1, inputs2],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True},
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_einsum_promotetype_mul(mode):
    """
    Feature: standard forward, backward features, test bmm in einsum.
    Description: test function einsum.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    class MulNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = mint.mul

        def construct(self, x, y):
            x = x.reshape(-1, 1)
            y = y.reshape(1, -1)
            return self.op(x, y)


    class MulGradNet(nn.Cell):
        def __init__(self, net):
            super(MulGradNet, self).__init__()
            self.grad = GradOperation(get_all=True, sens_param=False)
            self.net = net

        def construct(self, x, y):
            return self.grad(self.net)(x, y)

    x_np = np.random.randn(8,).astype(np.int64)
    y_np = np.random.randn(9,).astype(np.float32)
    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    equation = "i,j->ij"

    einsum_out = EinsumNet(equation)(x, y)
    einsum_grads = EinsumGradNet(EinsumNet(equation))(x, y)
    assert einsum_out.dtype == ms.float32
    np.testing.assert_allclose(einsum_out.asnumpy(), x_np.reshape(-1, 1) * y_np.reshape(1, -1), rtol=1e-4)

    mul_out = MulNet()(x, y)
    mul_grads = MulGradNet(MulNet())(x, y)
    assert einsum_out.dtype == mul_out.dtype
    assert einsum_grads[0].dtype == mul_grads[0].dtype
    assert einsum_grads[1].dtype == mul_grads[1].dtype
    np.testing.assert_allclose(einsum_out.asnumpy(), mul_out.asnumpy(), rtol=0)
    np.testing.assert_allclose(einsum_grads[0].asnumpy(), mul_grads[0].asnumpy(), rtol=0)
    np.testing.assert_allclose(einsum_grads[1].asnumpy(), mul_grads[1].asnumpy(), rtol=0)
