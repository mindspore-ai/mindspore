# Copyright 2026 Huawei Technologies Co., Ltd
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

"""DVM frontend boundary cases in pynative mode."""

import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import context
from mindspore import ops
from mindspore._c_expression import pyboost_detach
from mindspore.ops import stop_gradient

from tests.st.graph_kernel.gk_utils import compare_outputs, gen_flag, gen_input

np.random.seed(1)
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", pynative_synchronize=False)

REPEAT = 1


def _compare_case(flag, run_case, *inputs, cmp_precision=0.0):
    for i in range(REPEAT):
        outputs = run_case(*inputs)
        compare_outputs(gen_flag(flag, i), outputs, cmp_precision=cmp_precision)


def _contiguous_1d_inputs():
    x0 = gen_input((4097,), "float32").to("Ascend")
    x1 = gen_input((4097,), "float32").to("Ascend")
    return x0, x1


def _matrix_inputs(rows=257, cols=241):
    x0 = gen_input((rows, cols), "float32").to("Ascend")
    x1 = gen_input((rows, cols), "float32").to("Ascend")
    return x0, x1


def test_old_view_strided_slice():
    """
    Feature: DVM old op_graph strided_slice view boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0, x1 = _contiguous_1d_inputs()

    def run_case(x0_, x1_):
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = ops.strided_slice(y1, begin=(5,), end=(4091,), strides=(1,))
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_old_view_strided_slice", run_case, x0, x1)


def test_old_view_diagonal():
    """
    Feature: DVM old op_graph diagonal view boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0, x1 = _matrix_inputs(4097, 1)

    def run_case(x0_, x1_):
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = ops.diagonal(y1, offset=0, dim1=0, dim2=1)
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_old_view_diagonal", run_case, x0, x1)


def test_old_view_unstack():
    """
    Feature: DVM old op_graph unstack view boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0, x1 = _matrix_inputs(17, 4097)

    def run_case(x0_, x1_):
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = ops.unstack(y1, axis=0)[5]
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_old_view_unstack", run_case, x0, x1)


def test_vm_stop_gradient():
    """
    Feature: DVM VM stop_gradient boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0, x1 = _matrix_inputs()

    def run_case(x0_, x1_):
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = stop_gradient(y1)
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_vm_stop_gradient", run_case, x0, x1)


def test_vm_hook_backward():
    """
    Feature: DVM VM hook backward boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0, x1 = _matrix_inputs()

    def identity_hook(grad):
        return grad

    def run_case(x0_, x1_):
        hook = ops.HookBackward(identity_hook)
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = hook(y1)
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_vm_hook_backward", run_case, x0, x1)


def test_detach():
    """
    Feature: DVM detach boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0, x1 = _matrix_inputs()

    def run_case(x0_, x1_):
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = pyboost_detach(y1)
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_detach", run_case, x0, x1)


def test_tensor_getitem_int():
    """
    Feature: DVM tensor getitem int boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0, x1 = _matrix_inputs(17, 4097)

    def run_case(x0_, x1_):
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = y1[5]
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_tensor_getitem_int", run_case, x0, x1)


def test_tensor_getitem_slice():
    """
    Feature: DVM tensor getitem slice boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0, x1 = _contiguous_1d_inputs()

    def run_case(x0_, x1_):
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = y1[5:4091:1]
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_tensor_getitem_slice", run_case, x0, x1)


def test_tensor_getitem_bool():
    """
    Feature: DVM tensor getitem bool boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0, x1 = _matrix_inputs()
    mask = gen_input((257, 241), "bool").to("Ascend")

    def run_case(x0_, x1_, mask_):
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = y1[mask_]
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_tensor_getitem_bool", run_case, x0, x1, mask)


def test_tensor_getitem_advanced():
    """
    Feature: DVM tensor getitem advanced boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    rows = 257
    cols = 241
    index_size = 97
    x0, x1 = _matrix_inputs(rows, cols)
    row_idx = Tensor(((np.arange(index_size) * 3) % rows).astype(np.int64), ms.int64).to("Ascend")
    col_idx = Tensor(((np.arange(index_size) * 5) % cols).astype(np.int64), ms.int64).to("Ascend")

    def run_case(x0_, x1_, row_idx_, col_idx_):
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = y1[row_idx_, col_idx_]
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_tensor_getitem_advanced", run_case, x0, x1, row_idx, col_idx)


def test_tensor_setitem_int():
    """
    Feature: DVM tensor setitem int boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0, x1 = _matrix_inputs(17, 4097)
    value = gen_input((4097,), "float32").to("Ascend")

    def run_case(x0_, x1_, value_):
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = ops.add(y1, 0)
        y2[5] = value_
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_tensor_setitem_int", run_case, x0, x1, value)


def test_tensor_setitem_slice():
    """
    Feature: DVM tensor setitem slice boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0, x1 = _contiguous_1d_inputs()
    value = gen_input((4091 - 5,), "float32").to("Ascend")

    def run_case(x0_, x1_, value_):
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = ops.add(y1, 0)
        y2[5:4091:1] = value_
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_tensor_setitem_slice", run_case, x0, x1, value)


def test_tensor_setitem_bool():
    """
    Feature: DVM tensor setitem bool boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0, x1 = _matrix_inputs()
    mask = gen_input((257, 241), "bool").to("Ascend")
    value = gen_input((int(mask.asnumpy().sum()),), "float32").to("Ascend")

    def run_case(x0_, x1_, mask_, value_):
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = ops.add(y1, 0)
        y2[mask_] = value_
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_tensor_setitem_bool", run_case, x0, x1, mask, value)


def test_tensor_setitem_advanced():
    """
    Feature: DVM tensor setitem advanced boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    rows = 257
    cols = 241
    index_size = 97
    x0, x1 = _matrix_inputs(rows, cols)
    row_idx = Tensor(((np.arange(index_size) * 3) % rows).astype(np.int64), ms.int64).to("Ascend")
    col_idx = Tensor(((np.arange(index_size) * 5) % cols).astype(np.int64), ms.int64).to("Ascend")
    value = gen_input((index_size,), "float32").to("Ascend")

    def run_case(x0_, x1_, row_idx_, col_idx_, value_):
        y0 = ops.neg(x0_)
        y1 = ops.mul(y0, x1_)
        y2 = ops.add(y1, 0)
        y2[row_idx_, col_idx_] = value_
        y3 = ops.abs(y2)
        return ops.neg(y3)

    _compare_case("test_tensor_setitem_advanced", run_case, x0, x1, row_idx, col_idx, value)


def test_runtime_event():
    """
    Feature: DVM runtime event boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0, x1 = _contiguous_1d_inputs()
    stream1 = ms.runtime.Stream()
    stream2 = ms.runtime.Stream()
    event = ms.runtime.Event()

    def run_case(x0_, x1_, stream1_, stream2_, event_):
        with ms.runtime.StreamCtx(stream1_):
            y0 = ops.neg(x0_)
            y1 = ops.mul(y0, x1_)
            event_.record()

        with ms.runtime.StreamCtx(stream2_):
            event_.wait()
            y2 = ops.abs(y1)
            return ops.neg(y2)

    _compare_case("test_runtime_event", run_case, x0, x1, stream1, stream2, event)


def test_set_stream_limit():
    """
    Feature: DVM stream limit boundary
    Description: pynative mode
    Expectation: the result matches with the expected result
    """
    x0 = Tensor(np.full((4097,), 2.0, dtype=np.float32)).to("Ascend")
    x1 = Tensor(np.full((4097,), 3.0, dtype=np.float32)).to("Ascend")

    def run_case(x0_, x1_):
        stream_ = ms.runtime.Stream()
        ms.runtime.set_stream_limit(stream_, 4, 8)
        with ms.runtime.StreamCtx(stream_):
            y0 = ops.neg(x0_)
            y1 = ops.mul(y0, x1_)
            return y1

    _compare_case("test_set_stream_limit", run_case, x0, x1)
