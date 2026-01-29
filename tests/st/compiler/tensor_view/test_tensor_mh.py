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
def mh_forward_method(x):
    return x.mH

@jit(backend="ms_backend")
def mh_forward_class(x):
    return Tensor.adjoint(x)


@jit(backend="ms_backend")
def mh_backward_method(x, sens):
    return ms.grad(mh_forward_method, grad_position=0, sens_param=True)(x, sens)


@jit(backend="ms_backend")
def mh_backward_class(x, sens):
    return ms.grad(mh_forward_class, grad_position=0, sens_param=True)(x, sens)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048)
@pytest.mark.parametrize('data_type', [mstype.float16, mstype.float32, mstype.float64,
                                       mstype.complex64, mstype.complex128])
@pytest.mark.parametrize('method_type', ['method', 'class'])
def test_tensor_mh_different_type(data_type, method_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor mH api with view operator.
    Expectation: Run success and expected to use the TransposeExtView operator.
    """
    if data_type in (mstype.complex64, mstype.complex128):
        x_np = np.array([[0. + 0.j, 1. + 1.j], [2. + 2.j, 3. + 3.j]])
    else:
        x_np = np.random.randn(2, 3).astype(np.float32)
    x = Tensor(x_np, dtype=data_type)
    tester = TensorViewTestModule("mH", [x])
    if method_type == 'method':
        tester.forward_cmp(mh_forward_method)
        tester.backward_cmp(mh_backward_method)
    else:
        tester.forward_cmp(mh_forward_class)
        tester.backward_cmp(mh_backward_class)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048)
@pytest.mark.parametrize('shape', [(2, 3), (2, 3, 4), (3, 4, 5, 6)])
@pytest.mark.parametrize('method_type', ['method', 'class'])
def test_tensor_mh_different_dimensions(shape, method_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor mH api with view operator.
    Expectation: Run success and expected to use the TransposeExtView operator.
    """
    x = Tensor(np.random.randn(*shape), dtype=ms.float32)
    tester = TensorViewTestModule("mH", [x])
    if method_type == 'method':
        tester.forward_cmp(mh_forward_method)
        tester.backward_cmp(mh_backward_method)
    else:
        tester.forward_cmp(mh_forward_class)
        tester.backward_cmp(mh_backward_class)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048)
@pytest.mark.parametrize('val', [np.nan, np.inf])
@pytest.mark.parametrize('method_type', ['method', 'class'])
def test_tensor_mh_inf_nan(val, method_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor mH api with view operator with inf/nan.
    Expectation: Run success and expected to use the TransposeExtView operator.
    """
    x = Tensor(np.full((2, 3, 4), val), dtype=ms.float32)
    tester = TensorViewTestModule("mH", [x])
    if method_type == 'method':
        tester.forward_cmp(mh_forward_method)
        tester.backward_cmp(mh_backward_method)
    else:
        tester.forward_cmp(mh_forward_class)
        tester.backward_cmp(mh_backward_class)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048)
def test_tensor_mh_ir_view():
    """
    Feature: Support tensor view api.
    Description: Support tensor mH api with view operator.
    Expectation: Run success and expected to use the TransposeExtView operator.
    """
    dir_path = 'ir_mh'
    target_op_name = 'TransposeExtView'
    ms.context.set_context(save_graphs=True, save_graphs_path=dir_path)
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'

    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

    x1 = Tensor(np.random.randn(2, 3), dtype=mstype.float32)
    mh_forward_method(x1)
    check_ir_view_op(dir_path, target_op_name)
    mh_forward_class(x1)
    check_ir_view_op(dir_path, target_op_name)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    ms.context.set_context(save_graphs=False)
