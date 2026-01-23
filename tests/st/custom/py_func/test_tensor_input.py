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
Test custom py_func with tensor inputs.
"""
import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore.graph.api import jit
from mindspore.ops import _ms_pyfunc
from mindspore import Tensor
from tests.mark_utils import arg_mark


def infer_func(x, y):
    return x


@_ms_pyfunc(infer_func=infer_func)
def py_func_tensor_single(x: Tensor, y: Tensor) -> Tensor:
    return ops.Add()(x, y)


@_ms_pyfunc(infer_func=infer_func)
def py_func_tensor_complex(x: Tensor, y: Tensor) -> Tensor:
    result = ops.Add()(x, y)
    result = ops.Add()(result, y)
    result = ops.Add()(result, y)
    return result


class CustomNet(ms.nn.Cell):
    @jit
    def construct(self, x, y):
        return py_func_tensor_single(x, y)


class CustomNet1_1(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = ops.Add()

    @jit
    def construct(self, x, y):
        res = py_func_tensor_single(x, y)
        return self.add(res, y)


class CustomNet1_2(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = ops.Add()

    @jit
    def construct(self, x, y):
        out = self.add(x, x)
        res = py_func_tensor_single(out, y)
        return res


class CustomNet2(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = ops.Add()

    @jit
    def construct(self, x, y):
        out = self.add(x, x)
        res = py_func_tensor_single(x, y)
        return out, res


class CustomNet3(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = ops.Add()

    @jit
    def construct(self, x, y):
        res = py_func_tensor_single(x, y)
        out = self.add(res, x)
        return out, res


class CustomNet4(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = ops.Add()

    @jit
    def construct(self, x, y):
        out = self.add(x, y)
        out = py_func_tensor_complex(out, y)
        out = self.add(out, y)
        return out

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_py_func_tensor():
    """
    Feature: custom operator callback to Python function
    Description: test tensor inputs with single/complex operations
    Expectation: nn result matches numpy result
    """
    ms.set_device("Ascend")
    ms.set_context(save_graphs=False, save_graphs_path="./graphs")

    x = ms.Tensor(np.array([1, 2, 3], dtype=np.float16))
    y = ms.Tensor(np.array([4, 5, 6], dtype=np.float16))

    out = CustomNet()(x, y)
    expect = np.array([5, 7, 9])
    assert np.allclose(out.asnumpy(), expect)

    out = CustomNet1_1()(x, y)
    expect = np.array([9, 12, 15])
    assert np.allclose(out.asnumpy(), expect)

    out = CustomNet1_2()(x, y)
    expect = np.array([6, 9, 12])
    assert np.allclose(out.asnumpy(), expect)

    out = CustomNet2()(x, y)
    expect0 = np.array([2, 4, 6])
    expect1 = np.array([5, 7, 9])
    assert np.allclose(out[0].asnumpy(), expect0)
    assert np.allclose(out[1].asnumpy(), expect1)

    out = CustomNet3()(x, y)
    expect0 = np.array([6, 9, 12])
    expect1 = np.array([5, 7, 9])
    assert np.allclose(out[0].asnumpy(), expect0)
    assert np.allclose(out[1].asnumpy(), expect1)

    out = CustomNet4()(x, y)
    expect = np.array([21, 27, 33])
    assert np.allclose(out.asnumpy(), expect)
