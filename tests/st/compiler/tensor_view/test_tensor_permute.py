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
def permute_forward_method(x, input_perm):
    return x.permute(input_perm)


@jit(backend="ms_backend")
def permute_forward_class(x, input_perm):
    return Tensor.permute(x, input_perm)


@jit(backend="ms_backend")
def permute_backward_method(x, input_perm, sens):
    return ms.grad(permute_forward_method, grad_position=0, sens_param=True)(x, input_perm, sens)


@jit(backend="ms_backend")
def permute_backward_class(x, input_perm, sens):
    return ms.grad(permute_forward_class, grad_position=0, sens_param=True)(x, input_perm, sens)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048)
@pytest.mark.parametrize('method_type', ['method', 'class'])
@pytest.mark.parametrize('data_type', [mstype.float16, mstype.float32, mstype.float64,
                                       mstype.int16, mstype.int32, mstype.int64])
def test_tensor_permute_different_type(method_type, data_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor permute api with view operator.
    Expectation: Run success and expected to use the TransposeExtView operator.
    """
    x = Tensor(np.arange(12).reshape(2, 2, 3), dtype=data_type)
    input_perm = (0, 2, 1)
    tester = TensorViewTestModule("permute", [x, input_perm])

    if method_type == 'method':
        tester.forward_cmp(permute_forward_method)
        tester.backward_cmp(permute_backward_method)
    else:
        tester.forward_cmp(permute_forward_class)
        tester.backward_cmp(permute_backward_class)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048)
@pytest.mark.parametrize('method_type', ['method', 'class'])
@pytest.mark.parametrize('shape_perm', [
    ((2, 2, 3), (0, 2, 1)),
    ((2, 3, 4, 5), (2, 0, 3, 1)),
])
def test_tensor_permute_different_dimensions(method_type, shape_perm):
    """
    Feature: Support tensor view api.
    Description: Support tensor permute api with view operator.
    Expectation: Run success and expected to use the TransposeExtView operator.
    """
    shape, input_perm = shape_perm
    x = Tensor(np.random.randn(*shape), dtype=ms.float32)
    tester = TensorViewTestModule("permute", [x, input_perm])

    if method_type == 'method':
        tester.forward_cmp(permute_forward_method)
        tester.backward_cmp(permute_backward_method)
    else:
        tester.forward_cmp(permute_forward_class)
        tester.backward_cmp(permute_backward_class)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048)
@pytest.mark.parametrize('val', [np.nan, np.inf])
@pytest.mark.parametrize('method_type', ['method', 'class'])
def test_tensor_permute_inf_nan(val, method_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor permute api with view operator with inf/nan.
    Expectation: Run success and expected to use the TransposeExtView operator.
    """
    x = Tensor(np.full((2, 2, 3), val), dtype=ms.float32)
    input_perm = (0, 2, 1)
    tester = TensorViewTestModule("permute", [x, input_perm])

    if method_type == 'method':
        tester.forward_cmp(permute_forward_method)
        tester.backward_cmp(permute_backward_method)
    else:
        tester.forward_cmp(permute_forward_class)
        tester.backward_cmp(permute_backward_class)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048)
def test_tensor_permute_ir_view():
    """
    Feature: Support tensor view api.
    Description: Support tensor permute api with view operator.
    Expectation: Run success and expected to use the TransposeExtView operator.
    """
    dir_path = 'ir_permute'
    target_op_name = 'TransposeExtView'
    ms.context.set_context(save_graphs=True, save_graphs_path=dir_path)
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'

    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

    x1 = Tensor(np.random.randn(2, 2, 3), dtype=mstype.float32)
    x2 = Tensor(np.random.randn(2, 3, 4, 5), dtype=mstype.float32)

    permute_forward_method(x1, (0, 2, 1))
    check_ir_view_op(dir_path, target_op_name)
    permute_forward_class(x2, (2, 0, 3, 1))
    check_ir_view_op(dir_path, target_op_name)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    ms.context.set_context(save_graphs=False)
