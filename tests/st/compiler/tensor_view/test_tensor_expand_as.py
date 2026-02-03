# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Test tensor view."""
import os
import shutil
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, jit
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark
from view_test_utils import TensorViewTestModule, check_ir_view_op


@jit(backend="ms_backend")
def expand_as_forward_method(x, other):
    return x.expand_as(other)


@jit(backend="ms_backend")
def expand_as_forward_class(x, other):
    return Tensor.expand_as(x, other)


@jit(backend="ms_backend")
def expand_as_backward_method(x, other, sens):
    return ms.grad(expand_as_forward_method, grad_position=0, sens_param=True)(x, other, sens)


@jit(backend="ms_backend")
def expand_as_backward_class(x, other, sens):
    return ms.grad(expand_as_forward_class, grad_position=0, sens_param=True)(x, other, sens)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048)
@pytest.mark.parametrize('method_type', ['method', 'class'])
@pytest.mark.parametrize('data_type', [mstype.float16, mstype.float32, mstype.float64,
                                       mstype.int16, mstype.int32, mstype.int64])
def test_tensor_expand_as_different_type(method_type, data_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor expand_as api with view operator.
    Expectation: Run success and expected to use the BroadcastToView operator.
    """
    x = Tensor(np.arange(3).reshape(1, 3), dtype=data_type)
    other = Tensor(np.zeros((2, 3)), dtype=data_type)
    tester = TensorViewTestModule("expand_as", [x, other])

    if method_type == 'method':
        tester.forward_cmp(expand_as_forward_method)
        tester.backward_cmp(expand_as_backward_method)
    else:
        tester.forward_cmp(expand_as_forward_class)
        tester.backward_cmp(expand_as_backward_class)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048)
@pytest.mark.parametrize('method_type', ['method', 'class'])
@pytest.mark.parametrize('shape', [
    ((1,), (3,)),
    ((1, 3), (2, 3)),
    ((1, 1, 5), (3, 4, 5)),
    ((2, 1, 4, 1), (2, 3, 4, 5)),
])
def test_tensor_expand_as_different_dimensions(method_type, shape):
    """
    Feature: Support tensor view api.
    Description: Support tensor expand_as api with view operator.
    Expectation: Run success and expected to use the BroadcastToView operator.
    """
    x_shape, other_shape = shape
    x = Tensor(np.random.randn(*x_shape), dtype=ms.float32)
    other = Tensor(np.random.randn(*other_shape), dtype=ms.float32)
    tester = TensorViewTestModule("expand_as", [x, other])

    if method_type == 'method':
        tester.forward_cmp(expand_as_forward_method)
        tester.backward_cmp(expand_as_backward_method)
    else:
        tester.forward_cmp(expand_as_forward_class)
        tester.backward_cmp(expand_as_backward_class)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048)
@pytest.mark.parametrize('val', [np.nan, np.inf])
@pytest.mark.parametrize('method_type', ['method', 'class'])
def test_tensor_expand_as_inf_nan(val, method_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor expand_as api with view operator with inf/nan.
    Expectation: Run success and expected to use the BroadcastToView operator.
    """
    x = Tensor(np.full((1, 3), val), dtype=ms.float32)
    other = Tensor(np.random.randn(2, 3), dtype=ms.float32)
    tester = TensorViewTestModule("expand_as", [x, other])

    if method_type == 'method':
        tester.forward_cmp(expand_as_forward_method)
        tester.backward_cmp(expand_as_backward_method)
    else:
        tester.forward_cmp(expand_as_forward_class)
        tester.backward_cmp(expand_as_backward_class)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048)
def test_tensor_expand_as_ir_view():
    """
    Feature: Support tensor view api.
    Description: Support tensor expand_as api with view operator.
    Expectation: Run success and expected to use the BroadcastToView operator.
    """
    dir_path = 'ir_expand_as'
    target_op_name = 'BroadcastToView'
    ms.context.set_context(save_graphs=True, save_graphs_path=dir_path)
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'

    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

    x1 = Tensor(np.random.randn(1, 3), dtype=mstype.float32)
    other1 = Tensor(np.random.randn(2, 3), dtype=mstype.float32)
    x2 = Tensor(np.random.randn(1, 1, 5), dtype=mstype.float32)
    other2 = Tensor(np.random.randn(3, 4, 5), dtype=mstype.float32)

    expand_as_forward_method(x1, other1)
    check_ir_view_op(dir_path, target_op_name)

    expand_as_forward_class(x2, other2)
    check_ir_view_op(dir_path, target_op_name)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    ms.context.set_context(save_graphs=False)
