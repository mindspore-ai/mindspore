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
def unsqueeze_forward_method(x, dim):
    return x.unsqueeze(dim)

@jit(backend="ms_backend")
def unsqueeze_forward_class(x, dim):
    return Tensor.unsqueeze(x, dim)

@jit(backend="ms_backend")
def unsqueeze_backward_method(x, dim, sens):
    return ms.grad(unsqueeze_forward_method, grad_position=0, sens_param=True)(x, dim, sens)

@jit(backend="ms_backend")
def unsqueeze_backward_class(x, dim, sens):
    return ms.grad(unsqueeze_forward_class, grad_position=0, sens_param=True)(x, dim, sens)

# unsupported op [ExpandDimsView] on GPU
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2048)
@pytest.mark.parametrize('method_type', ['method', 'class'])
@pytest.mark.parametrize('data_type', [mstype.float16, mstype.float32, mstype.float64,
                                       mstype.int16, mstype.int32, mstype.int64])
def test_tensor_unsqueeze_different_type(method_type, data_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor unsqueeze api with view operator.
    Expectation: Run success and expected to use the ExpandDimsView operator.
    """
    x = Tensor(np.arange(6), dtype=data_type)
    dim = 1

    tester = TensorViewTestModule("unsqueeze", [x, dim])

    if method_type == 'method':
        tester.forward_cmp(unsqueeze_forward_method)
        tester.backward_cmp(unsqueeze_backward_method)
    else:
        tester.forward_cmp(unsqueeze_forward_class)
        tester.backward_cmp(unsqueeze_backward_class)

#aclnnInplaceCopy not support 8D input
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential', mem_peak=20248)
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
def test_tensor_unsqueeze_different_dimensions(shape, method_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor unsqueeze api with view operator.
    Expectation: Run success and expected to use the ExpandDimsView operator.
    """
    x = Tensor(np.random.randn(*shape), dtype=ms.float32)
    dim = 0

    tester = TensorViewTestModule("unsqueeze", [x, dim])
    if method_type == 'method':
        tester.forward_cmp(unsqueeze_forward_method)
        tester.backward_cmp(unsqueeze_backward_method)
    else:
        tester.forward_cmp(unsqueeze_forward_class)
        tester.backward_cmp(unsqueeze_backward_class)

@arg_mark(plat_marks=['cpu_linux', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048)
@pytest.mark.parametrize('val', [np.nan, np.inf])
@pytest.mark.parametrize('method_type', ['method', 'class'])
def test_tensor_unsqueeze_inf_nan(val, method_type):
    """
    Feature: Support tensor view api.
    Description: Support tensor unsqueeze api with view operator with inf/nan.
    Expectation: Run success and expected to use the ExpandDimsView operator.
    """
    x = Tensor(np.full((3, 3), val), dtype=ms.float32)
    dim = 0

    tester = TensorViewTestModule("unsqueeze", [x, dim])
    if method_type == 'method':
        tester.forward_cmp(unsqueeze_forward_method)
        tester.backward_cmp(unsqueeze_backward_method)
    else:
        tester.forward_cmp(unsqueeze_forward_class)
        tester.backward_cmp(unsqueeze_backward_class)

@arg_mark(plat_marks=['cpu_linux', 'platform_ascend', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2048)
def test_tensor_unsqueeze_ir_view():
    """
    Feature: Support tensor view api.
    Description: Support tensor unsqueeze api with view operator.
    Expectation: Run success and expected to use the ExpandDimsView operator.
    """
    dir_path = 'ir_unsqueeze'
    target_op_name = 'ExpandDimsView'
    ms.context.set_context(save_graphs=True, save_graphs_path=dir_path)
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'

    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

    x1 = Tensor(np.arange(1, 4, 5), dtype=mstype.float32)
    x2 = Tensor(np.arange(2, 7, 8), dtype=mstype.float32)
    dim = 1

    unsqueeze_forward_method(x1, dim)
    check_ir_view_op(dir_path, target_op_name)

    unsqueeze_forward_class(x2, dim)
    check_ir_view_op(dir_path, target_op_name)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
