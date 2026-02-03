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
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, jit
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark
from view_test_utils import TensorViewTestModule, check_ir_view_op


@jit(backend="ms_backend")
def narrow_forward_method(x, dim, start, length):
    return x.narrow(dim, start, length)


@jit(backend="ms_backend")
def narrow_forward_class(x, dim, start, length):
    return Tensor.narrow(x, dim, start, length)


@jit(backend="ms_backend")
def narrow_backward_method(x, dim, start, length, sens):
    return ms.grad(narrow_forward_method, grad_position=0, sens_param=True)(x, dim, start, length, sens)


@jit(backend="ms_backend")
def narrow_backward_class(x, dim, start, length, sens):
    return ms.grad(narrow_forward_class, grad_position=0, sens_param=True)(x, dim, start, length, sens)


# @arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
@arg_mark(plat_marks=['platform_gpu', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048)
@pytest.mark.parametrize('method_type', ['method', 'class'])
@pytest.mark.parametrize('data_type', [mstype.float16, mstype.float32, mstype.float64,
                                       mstype.int32, mstype.int64])
def test_tensor_narrow_different_type(method_type, data_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor narrow api with view operator.
    Expectation: Run success and expected to use the SliceExtView operator.
    """
    x = Tensor(np.arange(9).reshape(3, 3), dtype=data_type)
    dim = 0
    start = 0
    length = 2
    tester = TensorViewTestModule("narrow", [x, dim, start, length])

    if method_type == 'method':
        tester.forward_cmp(narrow_forward_method)
        tester.backward_cmp(narrow_backward_method)
    else:
        tester.forward_cmp(narrow_forward_class)
        tester.backward_cmp(narrow_backward_class)


# @arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
@arg_mark(plat_marks=['platform_gpu', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048)
@pytest.mark.parametrize('shape', [
    (3,),
    (2, 3),
    (3, 4, 5),
    (2, 3, 4, 5),
    (2, 2, 3, 4, 5),
    (2, 2, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2, 2),
])
@pytest.mark.parametrize('method_type', ['method', 'class'])
def test_tensor_narrow_different_dimensions(shape, method_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor narrow api with view operator.
    Expectation: Run success and expected to use the SliceExtView operator.
    """
    x = Tensor(np.random.randn(*shape), dtype=ms.float32)
    dim = 0
    start = 0
    length = 1
    tester = TensorViewTestModule("narrow", [x, dim, start, length])

    if method_type == 'method':
        tester.forward_cmp(narrow_forward_method)
        tester.backward_cmp(narrow_backward_method)
    else:
        tester.forward_cmp(narrow_forward_class)
        tester.backward_cmp(narrow_backward_class)


# @arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
@arg_mark(plat_marks=['platform_gpu', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048)
@pytest.mark.parametrize('val', [np.nan, np.inf])
@pytest.mark.parametrize('method_type', ['method', 'class'])
def test_tensor_narrow_inf_nan(val, method_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor narrow api with view operator with inf/nan.
    Expectation: Run success and expected to use the SliceExtView operator.
    """
    x = Tensor(np.full((3, 3), val), dtype=ms.float32)
    dim = 0
    start = 0
    length = 2
    tester = TensorViewTestModule("narrow", [x, dim, start, length])

    if method_type == 'method':
        tester.forward_cmp(narrow_forward_method)
        tester.backward_cmp(narrow_backward_method)
    else:
        tester.forward_cmp(narrow_forward_class)
        tester.backward_cmp(narrow_backward_class)


# # @arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
@arg_mark(plat_marks=['platform_gpu', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048)
def test_tensor_narrow_ir_view():
    """
    Feature: Support tensor view api.
    Description: Support tensor narrow api with view operator.
    Expectation: Run success and expected to use the SliceExtView operator.
    """
    dir_path = 'ir_narrow'
    target_op_name = 'SliceExtView'
    ms.context.set_context(save_graphs=True, save_graphs_path=dir_path)
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'

    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

    x1 = Tensor(np.random.randn(3, 3), dtype=mstype.float32)
    x2 = Tensor(np.random.randn(4, 5), dtype=mstype.float32)
    dim = 0
    start = 0
    length = 2

    narrow_forward_method(x1, dim, start, length)
    check_ir_view_op(dir_path, target_op_name)

    narrow_forward_class(x2, dim, start, length)
    check_ir_view_op(dir_path, target_op_name)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    ms.context.set_context(save_graphs=False)
