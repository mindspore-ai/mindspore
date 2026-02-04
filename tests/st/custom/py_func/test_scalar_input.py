# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Test custom py_func with scalar-type inputs.
"""
import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore.graph.api import jit
from mindspore.ops import _ms_pyfunc
from mindspore import Tensor
from tests.mark_utils import arg_mark


def infer_func_scalar(x, y, z):
    return x


def py_func_scalar_impl(x, y, z):
    result = ops.Add()(x, y)
    result = result + z
    return result


@_ms_pyfunc(infer_func=infer_func_scalar)
def py_func_type_int(x: Tensor, y: Tensor, z: int) -> Tensor:
    return py_func_scalar_impl(x, y, z)


@_ms_pyfunc(infer_func=infer_func_scalar)
def py_func_type_float(x: Tensor, y: Tensor, z: float) -> Tensor:
    return py_func_scalar_impl(x, y, z)


@_ms_pyfunc(infer_func=infer_func_scalar)
def py_func_type_bool(x: Tensor, y: Tensor, z: bool) -> Tensor:
    return py_func_scalar_impl(x, y, z)


class CustomNetTypeFloat(ms.nn.Cell):
    @jit(backend="ms_backend")
    def construct(self, x, y, z):
        return py_func_type_float(x, y, z)


class CustomNetTypeInt(ms.nn.Cell):
    @jit(backend="ms_backend")
    def construct(self, x, y, z):
        return py_func_type_int(x, y, z)


class CustomNetTypeBool(ms.nn.Cell):
    @jit(backend="ms_backend")
    def construct(self, x, y, z):
        return py_func_type_bool(x, y, z)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_py_func_type_scalar():
    """
    Feature: custom operator callback to Python function
    Description: test scalar-type inputs (int/float/bool)
    Expectation: nn result matches numpy result
    """
    ms.set_device("Ascend")
    ms.set_context(save_graphs=False, save_graphs_path="./graphs")

    x = ms.Tensor(np.array([1, 2, 3], dtype=np.float16))
    y = ms.Tensor(np.array([4, 5, 6], dtype=np.float16))

    out = CustomNetTypeInt()(x, y, 5)
    out = CustomNetTypeInt()(x, y, 5)
    expect = np.array([10, 12, 14])
    assert np.allclose(out.asnumpy(), expect)

    out = CustomNetTypeFloat()(x, y, 5.0)
    expect = np.array([10, 12, 14])
    assert np.allclose(out.asnumpy(), expect)

    out = CustomNetTypeBool()(x, y, False)
    expect = np.array([5, 7, 9])
    assert np.allclose(out.asnumpy(), expect)

    out = CustomNetTypeBool()(x, y, True)
    expect = np.array([6, 8, 10])
    assert np.allclose(out.asnumpy(), expect)
