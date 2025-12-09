# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test the feature of functional"""
import pytest
import numpy as np
import mindspore as ms
from tests.mark_utils import arg_mark
from ..share.utils import assert_executed_by_graph_mode


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_tensor():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit(capture_mode="bytecode")
    def func_clamp_tensor(x, min, max):
      return x.clamp(min, max)

    x = ms.Tensor([1, 2, 3, 4, 5])
    out = func_clamp_tensor(x, ms.Tensor(2), ms.Tensor(4))
    assert np.all(out.asnumpy() == ms.Tensor([2, 2, 3, 4, 4]).asnumpy())
    assert_executed_by_graph_mode(func_clamp_tensor)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_scalar():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit(capture_mode="bytecode")
    def func_clamp_scalar(x, min, max):
      return x.clamp(min, max)

    x = ms.Tensor([1, 2, 3, 4, 5])
    out = func_clamp_scalar(x, 2, 4)
    assert np.all(out.asnumpy() == ms.Tensor([2, 2, 3, 4, 4]).asnumpy())
    assert_executed_by_graph_mode(func_clamp_scalar)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_default():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit(capture_mode="bytecode")
    def func(x, min):
      return x.clamp(min)

    out = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2))
    assert np.all(out.asnumpy() == ms.Tensor([2, 2, 3, 4, 5]).asnumpy())
    assert_executed_by_graph_mode(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_keyword():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit(capture_mode="bytecode")
    def func(x, min):
      return x.clamp(min, max=ms.Tensor(4)), x.clamp(max=3, min=1)

    out_clamp_tensor, out_clamp_scalar = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2))
    assert np.all(out_clamp_tensor.asnumpy() == ms.Tensor([2, 2, 3, 4, 4]).asnumpy())
    assert np.all(out_clamp_scalar.asnumpy() == ms.Tensor([1, 2, 3, 3, 3]).asnumpy())
    assert_executed_by_graph_mode(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_asnumpy():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit(capture_mode="bytecode")
    def func(x, min, max):
      return x.clamp(min, ms.Tensor(max.asnumpy()))

    out = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2), ms.Tensor(4))
    assert np.all(out.asnumpy() == ms.Tensor([2, 2, 3, 4, 4]).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_method_clamp_keyword_asnumpy():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.clamp.
    Expectation: Run success
    """
    @ms.jit(capture_mode="bytecode")
    def func(x, min, max):
      return x.clamp(min=ms.Tensor(min.asnumpy())), x.clamp(2, max=int(max.asnumpy()))

    out_clamp_tensor, out_clamp_scalar = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2), ms.Tensor(4))
    assert np.all(out_clamp_tensor.asnumpy() == ms.Tensor([2, 2, 3, 4, 5]).asnumpy())
    assert np.all(out_clamp_scalar.asnumpy() == ms.Tensor([2, 2, 3, 4, 4]).asnumpy())
