# Copyright 2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P
from mindspore import dtype as mstype


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_conj():
    """
    Feature: ALL TO ALL
    Description:  test cases for conj in graph mode cpu backend.
    Expectation: the result match numpy conj
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.asarray(np.complex(1.3 + 0.4j), dtype=np.complex64)
    ms_x = Tensor(x, mstype.complex64)
    output = P.Conj()(ms_x)
    expect = np.conj(x)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pynative_conj():
    """
    Feature: ALL TO ALL
    Description:  test cases for conj in pynative mode cpu backend.
    Expectation: the result match numpy conj
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = np.asarray(np.complex(1.3 + 0.4j), dtype=np.complex64)
    ms_x = Tensor(x, mstype.complex64)
    output = P.Conj()(ms_x)
    expect = np.conj(x)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_real():
    """
    Feature: ALL TO ALL
    Description:  test cases for real in graph mode cpu backend.
    Expectation: the result match numpy real
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.asarray(np.complex(1.3 + 0.4j), dtype=np.complex64)
    ms_x = Tensor(x, mstype.complex64)
    output = P.Real()(ms_x)
    expect = np.real(x)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pynative_real():
    """
    Feature: ALL TO ALL
    Description:  test cases for real in pynative mode cpu backend.
    Expectation: the result match numpy real
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.asarray(np.complex(1.3 + 0.4j), dtype=np.complex64)
    ms_x = Tensor(x, mstype.complex64)
    output = P.Real()(ms_x)
    expect = np.real(x)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_imag():
    """
    Feature: ALL TO ALL
    Description:  test cases for image in graph mode cpu backend.
    Expectation: the result match numpy conj
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.asarray(np.complex(1.3 + 0.4j), dtype=np.complex64)
    ms_x = Tensor(x, mstype.complex64)
    output = P.Imag()(ms_x)
    expect = np.imag(x)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pynative_imag():
    """
    Feature: ALL TO ALL
    Description:  test cases for image in pynative mode cpu backend.
    Expectation: the result match numpy image
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.asarray(np.complex(1.3 + 0.4j), dtype=np.complex64)
    ms_x = Tensor(x, mstype.complex64)
    output = P.Imag()(ms_x)
    expect = np.imag(x)
    assert np.allclose(output.asnumpy(), expect)





@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_ceil():
    """
    Feature: ALL TO ALL
    Description:  test cases for ceil in graph mode cpu backend.
    Expectation: the result match numpy ceil
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array([1.1, -2.1]).astype(np.float32))
    np_x = np.array([1.1, -2.1]).astype(np.float32)
    output = P.Ceil()(x)
    expect = np.ceil(np_x)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pynative_ceil():
    """
    Feature: ALL TO ALL
    Description:  test cases for ceil in pynative mode cpu backend.
    Expectation: the result match numpy ceil
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.array([1.1, -2.1]).astype(np.float32))
    np_x = np.array([1.1, -2.1]).astype(np.float32)
    output = P.Ceil()(x)
    expect = np.ceil(np_x)
    assert np.allclose(output.asnumpy(), expect)
