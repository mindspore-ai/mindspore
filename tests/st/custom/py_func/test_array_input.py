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
Test custom py_func with tuple/list scalar arrays.
"""
import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore.graph.api import jit
from mindspore.ops import _ms_pyfunc
from mindspore import Tensor
from typing import Tuple, List
from tests.mark_utils import arg_mark


def infer_func(x, y):
    return x[0]


def py_func_impl(x, y):
    new_y = ms.Tensor(np.array(y, np.float16))
    result = ops.AddN()(x)
    result = ops.Add()(result, new_y)
    return result


@_ms_pyfunc(infer_func=infer_func)
def py_func_tuple_int(x: Tuple[Tensor, ...], y: Tuple[int, ...]) -> Tensor:
    return py_func_impl(x, y)


@_ms_pyfunc(infer_func=infer_func)
def py_func_tuple_float(x: Tuple[Tensor, ...], y: Tuple[float, ...]) -> Tensor:
    return py_func_impl(x, y)


@_ms_pyfunc(infer_func=infer_func)
def py_func_tuple_bool(x: Tuple[Tensor, ...], y: Tuple[bool, ...]) -> Tensor:
    return py_func_impl(x, y)


@_ms_pyfunc(infer_func=infer_func)
def py_func_list_int(x: Tuple[Tensor, ...], y: List[int]) -> Tensor:
    return py_func_impl(x, y)


@_ms_pyfunc(infer_func=infer_func)
def py_func_list_float(x: Tuple[Tensor, ...], y: List[float]) -> Tensor:
    return py_func_impl(x, y)


@_ms_pyfunc(infer_func=infer_func)
def py_func_list_bool(x: Tuple[Tensor, ...], y: List[bool]) -> Tensor:
    return py_func_impl(x, y)


class CustomNetTupleInt(ms.nn.Cell):
    @jit(backend="ms_backend")
    def construct(self, x, y, z):
        return py_func_tuple_int((x, y), z)


class CustomNetTupleBool(ms.nn.Cell):
    @jit(backend="ms_backend")
    def construct(self, x, y, z):
        return py_func_tuple_bool((x, y), z)


class CustomNetTupleFloat(ms.nn.Cell):
    @jit(backend="ms_backend")
    def construct(self, x, y, z):
        return py_func_tuple_float((x, y), z)


class CustomNetListInt(ms.nn.Cell):
    @jit(backend="ms_backend")
    def construct(self, x, y, z):
        return py_func_list_int((x, y), z)


class CustomNetListBool(ms.nn.Cell):
    @jit(backend="ms_backend")
    def construct(self, x, y, z):
        return py_func_list_bool((x, y), z)


class CustomNetListFloat(ms.nn.Cell):
    @jit(backend="ms_backend")
    def construct(self, x, y, z):
        return py_func_list_float((x, y), z)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_py_func_array():
    """
    Feature: custom operator callback to Python function
    Description: test tuple/list scalar arrays as inputs
    Expectation: nn result matches numpy result
    """
    ms.set_device("Ascend")
    ms.set_context(save_graphs=False, save_graphs_path="./graphs")

    x = ms.Tensor(np.array([1, 2, 3], dtype=np.float16))
    y = ms.Tensor(np.array([4, 5, 6], dtype=np.float16))

    z = (2, 2, 2)
    expect = np.array([7, 9, 11])
    out = CustomNetTupleInt()(x, y, z)
    assert np.allclose(out.asnumpy(), expect)

    out = CustomNetListInt()(x, y, list(z))
    assert np.allclose(out.asnumpy(), expect)

    z = (2.5, 2.5, 2.5)
    expect = np.array([7.5, 9.5, 11.5])
    out = CustomNetTupleFloat()(x, y, z)
    assert np.allclose(out.asnumpy(), expect)

    out = CustomNetListFloat()(x, y, list(z))
    assert np.allclose(out.asnumpy(), expect)

    z = (False, True, True)
    expect = np.array([5, 8, 10])
    out = CustomNetTupleBool()(x, y, z)
    assert np.allclose(out.asnumpy(), expect)

    out = CustomNetListBool()(x, y, list(z))
    assert np.allclose(out.asnumpy(), expect)
