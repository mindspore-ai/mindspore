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
def test_scalar_add(mode):
    """
    Feature: test ScalarAdd.
    Description: inputs is mutable_scalar.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_add_forward_func(x, y):
        return F.scalar_add(x, y)

    @test_utils.run_with_cell
    def scalar_add_backward_func(x, y):
        return ops.grad(scalar_add_forward_func, (0, 1))(x, y)

    context.set_context(mode=mode)
    input_x = 3
    input_y = 4
    output = scalar_add_forward_func(input_x, input_y)
    expect_out = 7.
    assert np.allclose(output, expect_out)
    device_target = context.get_context("device_target")
    if mode == ms.PYNATIVE_MODE and device_target == "Ascend":
        # When input is mutable, InferValue dose not work. Since there is no backend implementation on Ascend and
        # heterogeneous computing is not supported in Graph Mode, this case will only run in Pynative Mode for Ascend.
        mutable_output = scalar_add_forward_func(ms.mutable(input_x), ms.mutable(input_y))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        # In Pynative Mode, scalar op will be computed with __call__ function, leading to constant folding. Thus, there
        # is no backward procedure in this case.
        setup_module()
        expect_grad_out = (1, 1)
        mutable_grad_output = scalar_add_backward_func(ms.mutable(input_x), ms.mutable(input_y))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_sub(mode):
    """
    Feature: test ScalarSub.
    Description: inputs is mutable_scalar.
    Expectation: the result match with numpy result
    """

    @test_utils.run_with_cell
    def scalar_sub_forward_func(x, y):
        return F.scalar_sub(x, y)

    @test_utils.run_with_cell
    def scalar_sub_backward_func(x, y):
        return ops.grad(scalar_sub_forward_func, (0, 1))(x, y)

    context.set_context(mode=mode)
    input_x = 3
    input_y = 4
    output = scalar_sub_forward_func(input_x, input_y)
    expect_out = -1
    assert np.allclose(output, expect_out)
    device_target = context.get_context("device_target")
    if mode == ms.PYNATIVE_MODE and device_target == "Ascend":
        mutable_output = scalar_sub_forward_func(ms.mutable(input_x), ms.mutable(input_y))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = (1, -1)
        mutable_grad_output = scalar_sub_backward_func(ms.mutable(input_x), ms.mutable(input_y))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_mul(mode):
    """
    Feature: test ScalarMul.
    Description: inputs is mutable_scalar.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_mul_forward_func(x, y):
        return F.scalar_mul(x, y)

    @test_utils.run_with_cell
    def scalar_mul_backward_func(x, y):
        return ops.grad(scalar_mul_forward_func, (0, 1))(x, y)

    context.set_context(mode=mode)
    input_x = 3
    input_y = 4
    output = scalar_mul_forward_func(input_x, input_y)
    expect_out = 12
    assert np.allclose(output, expect_out)
    device_target = context.get_context("device_target")
    if mode == ms.PYNATIVE_MODE and device_target == "Ascend":
        mutable_output = scalar_mul_forward_func(ms.mutable(input_x), ms.mutable(input_y))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = (4, 3)
        mutable_grad_output = scalar_mul_backward_func(ms.mutable(input_x), ms.mutable(input_y))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_div(mode):
    """
    Feature: test ScalarDiv.
    Description: inputs is mutable_scalar.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_div_forward_func(x, y):
        return F.scalar_div(x, y)

    @test_utils.run_with_cell
    def scalar_div_backward_func(x, y):
        return ops.grad(scalar_div_forward_func, (0, 1))(x, y)

    context.set_context(mode=mode)
    input_x = 3
    input_y = 4
    output = scalar_div_forward_func(input_x, input_y)
    expect_out = 3/4
    assert np.allclose(output, expect_out)
    device_target = context.get_context("device_target")
    if mode == ms.PYNATIVE_MODE and device_target == "Ascend":
        mutable_output = scalar_div_forward_func(ms.mutable(input_x), ms.mutable(input_y))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = (0.25, -0.1875)
        mutable_grad_output = scalar_div_backward_func(ms.mutable(input_x), ms.mutable(input_y))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_mod(mode):
    """
    Feature: test ScalarMod.
    Description: inputs is mutable_scalar. *Frontend optimization is required for Ascend case.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_mod_forward_func(x, y):
        return F.scalar_mod(x, y)

    @test_utils.run_with_cell
    def scalar_mod_backward_func(x, y):
        return ops.grad(scalar_mod_forward_func, (0, 1))(x, y)

    context.set_context(mode=mode)
    input_x = 3
    input_y = 4
    output = scalar_mod_forward_func(input_x, input_y)
    expect_out = 3
    assert np.allclose(output, expect_out)
    device_target = context.get_context("device_target")
    if mode == ms.PYNATIVE_MODE and device_target == "Ascend":
        mutable_output = scalar_mod_forward_func(ms.mutable(input_x), ms.mutable(input_y))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = (1, 0)
        mutable_grad_output = scalar_mod_backward_func(ms.mutable(input_x), ms.mutable(input_y))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_floordiv(mode):
    """
    Feature: test ScalarFloorDiv.
    Description: inputs is mutable_scalar. *Frontend optimization is required for Ascend case.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_floordiv_forward_func(x, y):
        return F.scalar_floordiv(x, y)

    @test_utils.run_with_cell
    def scalar_floordiv_backward_func(x, y):
        return ops.grad(scalar_floordiv_forward_func, (0, 1))(x, y)

    context.set_context(mode=mode)
    input_x = 3
    input_y = 4
    output = scalar_floordiv_forward_func(input_x, input_y)
    expect_out = 0
    assert np.allclose(output, expect_out)
    device_target = context.get_context("device_target")
    if mode == ms.PYNATIVE_MODE and device_target == "Ascend":
        mutable_output = scalar_floordiv_forward_func(ms.mutable(input_x), ms.mutable(input_y))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = (0, 0)
        mutable_grad_output = scalar_floordiv_backward_func(ms.mutable(input_x), ms.mutable(input_y))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_eq(mode):
    """
    Feature: test scalar_eq.
    Description: inputs is mutable_scalar.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_eq_forward_func(x, y):
        return F.scalar_eq(x, y)

    @test_utils.run_with_cell
    def scalar_eq_backward_func(x, y):
        return ops.grad(scalar_eq_forward_func, (0, 1))(x, y)

    context.set_context(mode=mode)
    input_x = 3
    input_y = 4
    output = scalar_eq_forward_func(input_x, input_y)
    expect_out = False
    assert np.allclose(output, expect_out)
    device_target = context.get_context("device_target")
    if mode == ms.PYNATIVE_MODE and device_target == "Ascend":
        mutable_output = scalar_eq_forward_func(ms.mutable(input_x), ms.mutable(input_y))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = (0, 0)
        mutable_grad_output = scalar_eq_backward_func(ms.mutable(input_x), ms.mutable(input_y))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_ge(mode):
    """
    Feature: test scalar_ge.
    Description: inputs is mutable_scalar.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_ge_forward_func(x, y):
        return F.scalar_ge(x, y)

    @test_utils.run_with_cell
    def scalar_ge_backward_func(x, y):
        return ops.grad(scalar_ge_forward_func, (0, 1))(x, y)

    context.set_context(mode=mode)
    input_x = 3
    input_y = 4
    output = scalar_ge_forward_func(input_x, input_y)
    expect_out = False
    assert np.allclose(output, expect_out)
    device_target = context.get_context("device_target")
    if mode == ms.PYNATIVE_MODE and device_target == "Ascend":
        mutable_output = scalar_ge_forward_func(ms.mutable(input_x), ms.mutable(input_y))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = (0, 0)
        mutable_grad_output = scalar_ge_backward_func(ms.mutable(input_x), ms.mutable(input_y))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_gt(mode):
    """
    Feature: test scalar_gt.
    Description: inputs is mutable_scalar.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_gt_forward_func(x, y):
        return F.scalar_gt(x, y)

    @test_utils.run_with_cell
    def scalar_gt_backward_func(x, y):
        return ops.grad(scalar_gt_forward_func, (0, 1))(x, y)

    context.set_context(mode=mode)
    input_x = 3
    input_y = 4
    output = scalar_gt_forward_func(input_x, input_y)
    expect_out = False
    assert np.allclose(output, expect_out)
    device_target = context.get_context("device_target")
    if mode == ms.PYNATIVE_MODE and device_target == "Ascend":
        mutable_output = scalar_gt_forward_func(ms.mutable(input_x), ms.mutable(input_y))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = (0, 0)
        mutable_grad_output = scalar_gt_backward_func(ms.mutable(input_x), ms.mutable(input_y))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_le(mode):
    """
    Feature: test scalar_le.
    Description: inputs is mutable_scalar.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_le_forward_func(x, y):
        return F.scalar_le(x, y)

    @test_utils.run_with_cell
    def scalar_le_backward_func(x, y):
        return ops.grad(scalar_le_forward_func, (0, 1))(x, y)

    context.set_context(mode=mode)
    input_x = 3
    input_y = 4
    output = scalar_le_forward_func(input_x, input_y)
    expect_out = True
    assert np.allclose(output, expect_out)
    device_target = context.get_context("device_target")
    if mode == ms.PYNATIVE_MODE and device_target == "Ascend":
        mutable_output = scalar_le_forward_func(ms.mutable(input_x), ms.mutable(input_y))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = (0, 0)
        mutable_grad_output = scalar_le_backward_func(ms.mutable(input_x), ms.mutable(input_y))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_lt(mode):
    """
    Feature: test scalar_lt.
    Description: inputs is mutable_scalar.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_lt_forward_func(x, y):
        return F.scalar_lt(x, y)

    @test_utils.run_with_cell
    def scalar_lt_backward_func(x, y):
        return ops.grad(scalar_lt_forward_func, (0, 1))(x, y)

    context.set_context(mode=mode)
    input_x = 3
    input_y = 4
    output = scalar_lt_forward_func(input_x, input_y)
    expect_out = True
    assert np.allclose(output, expect_out)
    device_target = context.get_context("device_target")
    if mode == ms.PYNATIVE_MODE and device_target == "Ascend":
        mutable_output = scalar_lt_forward_func(ms.mutable(input_x), ms.mutable(input_y))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = (0, 0)
        mutable_grad_output = scalar_lt_backward_func(ms.mutable(input_x), ms.mutable(input_y))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scalar_pow(mode):
    """
    Feature: test uadd.
    Description: inputs is mutable_scalar.
    Expectation: the result match with numpy result
    """
    @test_utils.run_with_cell
    def scalar_pow_forward_func(x, y):
        return F.scalar_pow(x, y)

    @test_utils.run_with_cell
    def scalar_pow_backward_func(x, y):
        return ops.grad(scalar_pow_forward_func, (0, 1))(x, y)

    context.set_context(mode=mode)
    input_x = 3
    input_y = 4
    output = scalar_pow_forward_func(input_x, input_y)
    expect_out = 81
    assert np.allclose(output, expect_out)
    device_target = context.get_context("device_target")
    if mode == ms.PYNATIVE_MODE and device_target == "Ascend":
        mutable_output = scalar_pow_forward_func(ms.mutable(input_x), ms.mutable(input_y))
        assert np.allclose(mutable_output, expect_out)
    if mode == ms.GRAPH_MODE:
        setup_module()
        expect_grad_out = (108, 88.9875946044)
        mutable_grad_output = scalar_pow_backward_func(ms.mutable(input_x), ms.mutable(input_y))
        teardown_module()
        assert np.allclose(mutable_grad_output, expect_grad_out)
