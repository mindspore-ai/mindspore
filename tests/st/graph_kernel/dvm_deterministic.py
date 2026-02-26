# Copyright 2026 Huawei Technologies Co., Ltd
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

"""
DVM deterministic test cases in pynative mode.
"""
import numpy as np
import mindspore
from mindspore import ops
from mindspore import context
from mindspore import mint
from tests.st.graph_kernel.gk_utils import gen_input

np.random.seed(1)
context.set_context(mode=context.PYNATIVE_MODE)
mindspore.set_deterministic(True)
repeat_num = 4


def _assert_repeat(get_output):
    base_output = None
    for _ in range(repeat_num):
        output = get_output()
        output = output.float().asnumpy() if output.dtype == mindspore.bfloat16 else output.asnumpy()
        if base_output is None:
            base_output = output
        else:
            np.testing.assert_allclose(base_output, output, 0, 0)


def _run_sum_ext():
    x0 = gen_input((32, 1024), "float32")
    for dims in [(0,), (1,), (0, 1)]:
        _assert_repeat(lambda dim=dims: ops.auto_generate.SumExt()(x0, dim, False))


def _run_matmul_ext():
    for data_type in ["float16", "bfloat16"]:
        x0 = gen_input((2, 4096, 1024), data_type)
        x1 = gen_input((1024, 2048), data_type)

        def get_output(a0=x0, a1=x1):
            return mint.matmul(a0, a1)

        _assert_repeat(get_output)


def _run_gmm_grouptype0():
    m, k, n = 320, 256, 128
    group_list = [0, 10, 30, 100, 140, 180, 220, 220, 240, 320]
    e = len(group_list)
    shape1 = (m, k)
    shape2 = (e, k, n)
    x = gen_input(shape1, "float16", data_range=0.01)
    weight = gen_input(shape2, "float16", data_range=0.01)
    bias = [gen_input((e, n), "float16", data_range=0.01)]
    group_list_ms = mindspore.Tensor(group_list, dtype=mindspore.int64)

    def get_output():
        op = ops.auto_generate.GroupedMatmul(3, 0, False, False)
        return op([x], [weight], bias, None, None, None, None, group_list_ms)[0]

    _assert_repeat(get_output)


def _run_gmm_grouptype2():
    m, k, n = 1024, 2560, 1024
    group_list = [0, 10, 300, 1024, 1024, 1400, 1800, 2048, 2048, 2200, 2560]
    shape1 = (k, m)
    shape2 = (k, n)
    x = gen_input(shape1, "bfloat16", data_range=0.01)
    weight = gen_input(shape2, "bfloat16", data_range=0.01)
    group_list_ms = mindspore.Tensor(group_list, dtype=mindspore.int64)

    def get_output():
        x0 = mint.transpose(x, -1, -2)
        op = ops.auto_generate.GroupedMatmul(3, 2, False, False)
        return op([x0], [weight], None, None, None, None, None, group_list_ms)[0]

    _assert_repeat(get_output)


def test_deterministics():
    """
    Feature: DVM deterministic
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    _run_sum_ext()
    _run_matmul_ext()
    _run_gmm_grouptype0()
    _run_gmm_grouptype2()
