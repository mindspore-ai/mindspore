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
# pylint: disable=unused-variable
import numpy as np
import pytest
from tests.st.utils import test_utils

from mindspore import ops
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def avg_pool_forward_func(x):
    return ops.AvgPool(kernel_size=2, strides=2, pad_mode="VALID", data_format="NCHW")(x)


@test_utils.run_with_cell
def avg_pool_backward_func(x):
    return ops.grad(avg_pool_forward_func, (0,))(x)


def avg_pool_dyn_shape_func(x):
    return ops.AvgPool(kernel_size=2, strides=2, pad_mode="VALID", data_format="NCHW")(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_avg_pool_forward(mode):
    """
    Feature: Ops.
    Description: test op avg pool.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    out = avg_pool_forward_func(x)
    expect = np.array(
        [[[[2.5, 4.5]], [[14.5, 16.5]], [[26.5, 28.5]]]]).astype(np.float32)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_avg_pool_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op avg pool.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    ms.context.set_context(precompile_only=True)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    grads = avg_pool_backward_func(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_avg_pool_vmap(mode):
    """
    Feature: test vmap function.
    Description: test avgpool op vmap.
    Expectation: expect correct result.
    """
    in_axes = -1
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4 * 2 * 1).reshape(1, 3, 3, 4, 2, 1).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(avg_pool_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x)
    expect = np.array(
        [[[[[[5., 9.]], [[29., 33.]], [[53., 57.]]]],
          [[[[6., 10.]], [[30., 34.]], [[54., 58.]]]],
          ]]).astype(np.float32)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_avg_pool_dynamic(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of avg pool.
    Description: test dynamic tensor and dynamic scalar of avg pool.
    Expectation: expect correct result.
    """
    # in1 = [ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32), 2, 2, "VALID", "NCHW"]
    # in2 = [ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32), 2, 2, "VALID", "NCHW"]
    # TEST_OP(ops.auto_generate.avg_pool, [in1, in2], dump_ir=True, grad=False)

    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    test_cell = test_utils.to_cell_obj(avg_pool_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    out = test_cell(x)
