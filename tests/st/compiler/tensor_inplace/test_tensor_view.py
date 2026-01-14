# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
""" test_tensor_view.py """
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor
from mindspore.ops.auto_generate.gen_ops_prim import BroadcastToView, ExpandDimsView, TransposeView
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_max():
    """
    Feature: Support view ops with keyword args input.
    Description: Support view ops with keyword args input.
    Expectation: Run success.
    """

    def tensor_max(tensor, axis=None, keepdims=False, initial=None, where=True):
        return tensor.max(axis, keepdims, initial=initial, where=where)

    @ms.jit(backend="ms_backend")
    def func():
        a = np.random.rand(2, 3).astype(np.float32)
        b = Tensor(a)
        where = np.random.randint(low=0, high=2,
                                  size=[2, 3]).astype(np.bool_)
        out_np = a.max(initial=2.0, where=where, axis=-1)
        out_ms = tensor_max(Tensor(b), initial=2.0,
                            where=Tensor(where), axis=-1)
        return out_np, out_ms.asnumpy()

    np_array, ms_array = func()
    np.allclose(np_array, ms_array, rtol=5e-03, atol=1.e-8)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_view_fallback_pyexecute():
    """
    Feature: The input of the view operator does not support non-continuous inputs.
    Description: Usage of View Operators in Heterogeneous Scenarios.
    Expectation: no exception
    """

    def func(x):
        x = ops.abs(x)
        view_obj1 = BroadcastToView()(x, (1, 4, 2))
        view_obj2 = ExpandDimsView()(view_obj1, 0)
        if x[0][0] > 0:
            view_obj2.mul_(2)
        else:
            view_obj2.mul_(3)
        return view_obj2, x

    def func_pyexecute(x):
        x = TransposeView()(Tensor(x.asnumpy()), (1, 0))
        _, y = func(x)
        y.mul_(x)
        return y

    with pytest.raises(RuntimeError) as err:
        x_np = np.ones([2, 4]).astype(np.float32)
        input_x = Tensor(x_np)
        func18_jit = ms.jit(func_pyexecute, backend="ms_backend")
        func18_jit(input_x)
    assert "Not support non-contiguous heter input" in str(err.value)
