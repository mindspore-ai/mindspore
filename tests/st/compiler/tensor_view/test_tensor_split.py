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
def split_forward_method(x, size, dim):
    return x.split(size, dim)

@jit(backend="ms_backend")
def split_forward_class(x, size, dim):
    return Tensor.split(x, size, dim)

@jit(backend="ms_backend")
def split_backward_method(x, size, dim, sens):
    return ms.grad(split_forward_method, grad_position=0, sens_param=True)(x, size, dim, sens)

@jit(backend="ms_backend")
def split_backward_class(x, size, dim, sens):
    return ms.grad(split_forward_class, grad_position=0, sens_param=True)(x, size, dim, sens)

# unsupported op [SplitTensorView] on GPU
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2048)
@pytest.mark.parametrize('method_type', ['method', 'class'])
@pytest.mark.parametrize('data_type', [mstype.float16, mstype.float32, mstype.float64,
                                       mstype.int16, mstype.int32, mstype.int64])
def test_tensor_split_different_type_int(method_type, data_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor split api (int size) with view operator.
    Expectation: Run success and expected to use the SplitTensorView operator.
    """
    x = Tensor(np.arange(9), dtype=data_type)
    size = 3
    dim = 0

    tester = TensorViewTestModule("split", [x, size, dim])

    if method_type == 'method':
        tester.forward_cmp(split_forward_method)
        tester.backward_cmp(split_backward_method)
    else:
        tester.forward_cmp(split_forward_class)
        tester.backward_cmp(split_backward_class)

# unsupported op [SplitWithSizeView] on GPU
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2048)
@pytest.mark.parametrize('method_type', ['method', 'class'])
@pytest.mark.parametrize('data_type', [mstype.float16, mstype.float32, mstype.float64,
                                       mstype.int16, mstype.int32, mstype.int64])
def test_tensor_split_different_type_list(method_type, data_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor split api (list size) with view operator.
    Expectation: Run success and expected to use the SplitWithSizeView operator.
    """
    x = Tensor(np.arange(10), dtype=data_type)
    size = [2, 3, 5]
    dim = 0

    tester = TensorViewTestModule("split", [x, size, dim])

    if method_type == 'method':
        tester.forward_cmp(split_forward_method)
        tester.backward_cmp(split_backward_method)
    else:
        tester.forward_cmp(split_forward_class)
        tester.backward_cmp(split_backward_class)

# unsupported op [SplitTensorView] on GPU
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048)
@pytest.mark.parametrize('shape', [
    (3,),
    (2, 3),
    (3, 4, 5),
    (2, 3, 4, 5),
    (2, 2, 3, 4, 5),
    (2, 2, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2, 2, 2),
])
@pytest.mark.parametrize('method_type', ['method', 'class'])
def test_tensor_split_different_dimensions_int(shape, method_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor split api with view operator.
    Expectation: Run success and expected to use the SplitTensorView operator.
    """
    x = Tensor(np.random.randn(*shape), dtype=ms.float32)
    size = 1
    dim = 0

    tester = TensorViewTestModule("split", [x, size, dim])
    if method_type == 'method':
        tester.forward_cmp(split_forward_method)
        tester.backward_cmp(split_backward_method)
    else:
        tester.forward_cmp(split_forward_class)
        tester.backward_cmp(split_backward_class)

# unsupported op [SplitWithSizeView] on GPU
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048)
@pytest.mark.parametrize('shape', [
    (3,),
    (3, 3),
    (3, 4, 5),
    (3, 3, 4, 5),
    (3, 2, 3, 4, 5),
    (3, 2, 2, 2, 2, 2),
    (3, 2, 2, 2, 2, 2, 2),
    (3, 2, 2, 2, 2, 2, 2, 2),
])
@pytest.mark.parametrize('method_type', ['method', 'class'])
def test_tensor_split_different_dimensions_list(shape, method_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor split api with view operator.
    Expectation: Run success and expected to use the SplitWithSizeView operator.
    """
    x = Tensor(np.random.randn(*shape), dtype=ms.float32)
    size = [1, 1, 1]
    dim = 0

    tester = TensorViewTestModule("split", [x, size, dim])
    if method_type == 'method':
        tester.forward_cmp(split_forward_method)
        tester.backward_cmp(split_backward_method)
    else:
        tester.forward_cmp(split_forward_class)
        tester.backward_cmp(split_backward_class)

@arg_mark(plat_marks=['cpu_linux', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048)
@pytest.mark.parametrize('val', [np.nan, np.inf])
@pytest.mark.parametrize('method_type', ['method', 'class'])
def test_tensor_split_inf_nan_int(val, method_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor split api (int size) with view operator with inf/nan.
    Expectation: Run success and expected to use the SplitTensorView operator.
    """
    x = Tensor(np.full((3, 3), val), dtype=ms.float32)
    size = 1
    dim = 0

    tester = TensorViewTestModule("split", [x, size, dim])
    if method_type == 'method':
        tester.forward_cmp(split_forward_method)
        tester.backward_cmp(split_backward_method)
    else:
        tester.forward_cmp(split_forward_class)
        tester.backward_cmp(split_backward_class)

@arg_mark(plat_marks=['cpu_linux', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048)
@pytest.mark.parametrize('val', [np.nan, np.inf])
@pytest.mark.parametrize('method_type', ['method', 'class'])
def test_tensor_split_inf_nan_list(val, method_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor split api (list size) with view operator with inf/nan.
    Expectation: Run success and expected to use the SplitWithSizeView operator.
    """
    x = Tensor(np.full((3, 3), val), dtype=ms.float32)
    size = [1, 1, 1]
    dim = 0

    tester = TensorViewTestModule("split", [x, size, dim])
    if method_type == 'method':
        tester.forward_cmp(split_forward_method)
        tester.backward_cmp(split_backward_method)
    else:
        tester.forward_cmp(split_forward_class)
        tester.backward_cmp(split_backward_class)

@arg_mark(plat_marks=['cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2048)
def test_tensor_split_ir_view_int():
    """
    Feature: Support tensor view api.
    Description: Support tensor split api (int size) with view operator.
    Expectation: Run success and expected to use the SplitTensorView operator.
    """
    dir_path = 'ir_split_int'
    target_op_name = 'SplitTensorView'
    ms.context.set_context(save_graphs=True, save_graphs_path=dir_path)
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'

    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

    x1 = Tensor(np.random.randn(4, 3, 2, 1), dtype=mstype.float32)
    x2 = Tensor(np.random.randn(6, 9, 3, 2), dtype=mstype.float32)
    size = 3
    dim = 0

    split_forward_method(x1, size, dim)
    check_ir_view_op(dir_path, target_op_name)

    split_forward_class(x2, size, dim)
    check_ir_view_op(dir_path, target_op_name)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')

@arg_mark(plat_marks=['cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2048)
def test_tensor_split_ir_view_list():
    """
    Feature: Support tensor view api.
    Description: Support tensor split api (list size) with view operator.
    Expectation: Run success and expected to use the SplitWithSizeView operator.
    """
    dir_path = 'ir_split_list'
    target_op_name = 'SplitWithSizeView'
    ms.context.set_context(save_graphs=True, save_graphs_path=dir_path)
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'

    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

    x1 = Tensor(np.random.randn(10, 4, 5), dtype=mstype.float32)
    x2 = Tensor(np.random.randn(20, 7, 8), dtype=mstype.float32)
    size = [2, 3, 5]
    size2 = [4, 6, 10]
    dim = 0

    split_forward_method(x1, size, dim)
    check_ir_view_op(dir_path, target_op_name)

    split_forward_class(x2, size2, dim)
    check_ir_view_op(dir_path, target_op_name)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
