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
# pylint: skip-file
"""ST tests for one_hot_ext."""

import numpy as np
import pytest

import mindspore as ms
from mindspore import ops
from mindspore.ops.function.array_func import one_hot_ext as one_hot
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase


def generate_random_input(num_classes):
    return np.random.permutation(num_classes)


def onehot_forward_func(tensor, num_classes):
    return one_hot(tensor, num_classes)


def call_onehot(tensor, num_classes):
    return ops.function.array_func.one_hot_ext(tensor, num_classes)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', mem_peak=2048.05)
@pytest.mark.parametrize('param_jit_level', ["O2", "O0"])
def test_onehot_vmap(param_jit_level):
    """
    Feature: Test onehot with vmap.
    Description: call one_hot with valid input and index.
    Expectation: return the correct value.
    """

    def _foreach_run(inputs, num_classes, batch):
        out = []
        for i in range(inputs.shape[batch]):
            input_inner = inputs[..., i] if batch == -1 else inputs[i, ...]
            out.append(onehot_forward_func(input_inner, num_classes))
        return ops.Stack()(out)

    ms.set_context(mode=ms.PYNATIVE_MODE, jit_level=param_jit_level)
    x = generate_random_input(2)

    for batch_axis in (-1, 0):
        onehot_ext_vmap_func = ops.vmap(call_onehot, in_axes=(batch_axis, None), out_axes=0)
        output = onehot_ext_vmap_func(ms.Tensor(x), 3)
        expect = _foreach_run(ms.Tensor(x), 3, batch_axis)
        assert np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-4)


def ops_one_hot_binary_compare(input_binary_data, output_binary_data, num_classes):
    output = onehot_forward_func(ms.Tensor(input_binary_data[0]), num_classes)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)


@ops_binary_cases(OpsBinaryCase(input_info=[((12, 80, 64), np.int64)], output_info=[((12, 80, 64, 33), np.int64)],
                                extra_info='auto_drive'))
def ops_one_hot_binary_case1(input_binary_data=None, output_binary_data=None):
    ops_one_hot_binary_compare(input_binary_data, output_binary_data, 33)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 16, 576, 128), np.int64)],
                                output_info=[((1, 16, 576, 128, 7), np.int64)],
                                extra_info='auto_drive'))
def ops_one_hot_binary_case2(input_binary_data=None, output_binary_data=None):
    ops_one_hot_binary_compare(input_binary_data, output_binary_data, 7)


@ops_binary_cases(OpsBinaryCase(input_info=[((4800, 16), np.int64)], output_info=[((4800, 16, 16), np.int64)],
                                extra_info='auto_drive'))
def ops_one_hot_binary_case3(input_binary_data=None, output_binary_data=None):
    ops_one_hot_binary_compare(input_binary_data, output_binary_data, 16)


@ops_binary_cases(OpsBinaryCase(input_info=[((8512,), np.int64)], output_info=[((8512, 118528), np.int64)],
                                extra_info='pg7B'))
def ops_one_hot_binary_case4(input_binary_data=None, output_binary_data=None):
    ops_one_hot_binary_compare(input_binary_data, output_binary_data, 118528)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 8192), np.int64)], output_info=[((1, 8192, 8), np.int64)],
                                extra_info='pgmoe'))
def ops_one_hot_binary_case5(input_binary_data=None, output_binary_data=None):
    ops_one_hot_binary_compare(input_binary_data, output_binary_data, 8)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', mem_peak=3072.0)
@pytest.mark.parametrize('mode', ["pynative", "kbk"])
def test_ops_onehot_binary_cases(mode):
    """
    Feature: pyboost function.
    Description: test function onehot forward with binary data.
    Expectation: expect correct result.
    """
    if mode == "kbk":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    ops_one_hot_binary_case1()
    ops_one_hot_binary_case2()
    ops_one_hot_binary_case3()
    ops_one_hot_binary_case5()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', mem_peak=9216.00)
def test_ops_onehot_huge_binary_cases():
    """
    Feature: pyboost function.
    Description: test function onehot forward with binary data.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    ops_one_hot_binary_case4()
