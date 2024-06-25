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
from mindspore import context
from mindspore import ops
import mindspore.ops.operations.manually_defined as F
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def setup_module():
    context.set_context(grad_for_scalar=True)


def teardown_module():
    context.set_context(grad_for_scalar=False)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_bool_mutable(mode):
    """
    Feature: test scalar bool.
    Description: inputs is mutable scalar.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_bool_forward_func(x):
        return F.scalar_bool(x)

    @test_utils.run_with_cell
    def scalar_bool_backward_func(x):
        return ops.grad(scalar_bool_forward_func, (0,))(x)

    context.set_context(mode=mode)
    input_x = 3
    output = scalar_bool_forward_func(input_x)
    expect_out = True
    assert np.allclose(output, expect_out)
    if mode == ms.PYNATIVE_MODE:
        mutable_output = scalar_bool_forward_func(ms.mutable(input_x))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        expect_grad_out = 0
        mutable_grad_output = scalar_bool_backward_func(ms.mutable(input_x))
        assert np.allclose(mutable_grad_output, expect_grad_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_uadd_mutable(mode):
    """
    Feature: test scalar uadd.
    Description: inputs is mutable scalar.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_uadd_forward_func(x):
        return F.scalar_uadd(x)

    @test_utils.run_with_cell
    def scalar_uadd_backward_func(x):
        return ops.grad(scalar_uadd_forward_func, (0,))(x)

    context.set_context(mode=mode)
    input_x = 3
    output = scalar_uadd_forward_func(input_x)
    expect_out = 3
    assert np.allclose(output, expect_out)
    if mode == ms.PYNATIVE_MODE:
        mutable_output = scalar_uadd_forward_func(ms.mutable(input_x))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = 1
        mutable_grad_output = scalar_uadd_backward_func(ms.mutable(input_x))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_usub_mutable(mode):
    """
    Feature: test scalar usub.
    Description: inputs is mutable scalar.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_usub_forward_func(x):
        return F.scalar_usub(x)

    @test_utils.run_with_cell
    def scalar_usub_backward_func(x):
        return ops.grad(scalar_usub_forward_func, (0,))(x)

    context.set_context(mode=mode)
    input_x = 3
    output = scalar_usub_forward_func(input_x)
    expect_out = -3
    assert np.allclose(output, expect_out)
    if mode == ms.PYNATIVE_MODE:
        mutable_output = scalar_usub_forward_func(ms.mutable(input_x))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = -1
        mutable_grad_output = scalar_usub_backward_func(ms.mutable(input_x))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_log_mutable(mode):
    """
    Feature: test scalar log.
    Description: inputs is mutable scalar.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_log_forward_func(x):
        return F.scalar_log(x)

    @test_utils.run_with_cell
    def scalar_log_backward_func(x):
        return ops.grad(scalar_log_forward_func, (0,))(x)

    context.set_context(mode=mode)
    input_x = 3
    output = scalar_log_forward_func(input_x)
    expect_out = 1.0986123
    assert np.allclose(output, expect_out)
    if mode == ms.PYNATIVE_MODE:
        mutable_output = scalar_log_forward_func(ms.mutable(input_x))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = 0.333333343267
        mutable_grad_output = scalar_log_backward_func(ms.mutable(input_x))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)
