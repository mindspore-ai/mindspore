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
Test custom py_func with multi tensor inputs/outputs.
"""
import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore.graph.api import jit
from mindspore.ops import _ms_pyfunc
from mindspore import Tensor
from typing import Tuple, List
from tests.mark_utils import arg_mark

def infer_func_multi_input(x, y):
    return x[0]


def infer_func_multi_output(x, y):
    return x[0], y


@_ms_pyfunc(infer_func=infer_func_multi_input)
def py_func_multi_input_tuple(x: Tuple[Tensor], y: Tensor) -> Tensor:
    result = ops.AddN()(x)
    result = ops.Add()(result, y)
    return result


@_ms_pyfunc(infer_func=infer_func_multi_input)
def py_func_multi_input_list(x: List[Tensor], y: Tensor) -> Tensor:
    result = ops.AddN()(x)
    result = ops.Add()(result, y)
    return result


@_ms_pyfunc(infer_func=infer_func_multi_output)
def py_func_multi_output(x: Tuple[Tensor], y: Tensor) -> Tuple[Tensor]:
    result1 = ops.AddN()(x)
    result2 = ops.Add()(result1, y)
    return result1, result2


class CustomNetTupleInput(ms.nn.Cell):
    @jit(backend="ms_backend")
    def construct(self, x, y):
        return py_func_multi_input_tuple((x, y), x)


class CustomNetListInput(ms.nn.Cell):
    @jit(backend="ms_backend")
    def construct(self, x, y):
        return py_func_multi_input_list([x, y], x)


class CustomNetTupleOutput(ms.nn.Cell):
    @jit(backend="ms_backend")
    def construct(self, x, y):
        return py_func_multi_output((x, y), x)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_py_func_multi_tensor():
    """
    Feature: custom operator callback to Python function
    Description: test multi tensor inputs/outputs with tuple/list
    Expectation: nn result matches numpy result
    """
    ms.set_device("Ascend")
    ms.set_context(save_graphs=False, save_graphs_path="./graphs")

    x = ms.Tensor(np.array([1, 2, 3], dtype=np.float16))
    y = ms.Tensor(np.array([4, 5, 6], dtype=np.float16))

    out = CustomNetTupleInput()(x, y)
    expect = np.array([6, 9, 12])
    assert np.allclose(out.asnumpy(), expect)

    out = CustomNetListInput()(x, y)
    expect = np.array([6, 9, 12])
    assert np.allclose(out.asnumpy(), expect)

    out = CustomNetTupleOutput()(x, y)
    expect0 = np.array([5, 7, 9])
    expect1 = np.array([6, 9, 12])
    assert np.allclose(out[0].asnumpy(), expect0)
    assert np.allclose(out[1].asnumpy(), expect1)
