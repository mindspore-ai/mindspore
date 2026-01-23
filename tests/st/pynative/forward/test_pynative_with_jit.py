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
import numpy as np
import mindspore
from mindspore import Tensor, ops, jit, mint
from tests.mark_utils import arg_mark

@jit
def func_jit(x):
    return ops.sin(x)

def func(x):
    return ops.sin(x)

def test_pynative_with_jit():
    """
    Feature: PyNative with jit.
    Description: Test running PyNative with jit.
    Expectation: run success
    """
    def test_func(input_np, run_func):
        for _ in range(10):
            x = Tensor(input_np)
            y = Tensor(input_np)

            for _ in range(1000):
                out = mint.matmul(x, y)
            out = run_func(out)
            assert np.allclose(out.asnumpy(), np.sin(np.matmul(input_np, input_np)))

    input_np = np.ones((1024, 1024)).astype(np.float32)

    test_func(input_np, func)
    test_func(input_np, func_jit)


def test_multi_stream_with_jit():
    """
    Feature: PyNative with jit.
    Description: Test running PyNative multi-stream wait before jit.
    Expectation: run success
    """
    input_np = np.ones((1024, 1024)).astype(np.float32)
    s1 = mindspore.hal.Stream()
    for _ in range(10):
        x = Tensor(input_np)
        y = Tensor(input_np)

        with mindspore.hal.StreamCtx(s1):
            for _ in range(1000):
                out = mint.matmul(x, y)
            event = s1.record_event()

        cur_stream = mindspore.hal.current_stream()
        cur_stream.wait_event(event)

        out = func_jit(out)
        assert np.allclose(out.asnumpy(), np.sin(np.matmul(input_np, input_np)))

def test_multi_stream_with_event():
    """
    Feature: PyNative multi-stream.
    Description: Test running PyNative with stream/event.
    Expectation: run success
    """
    input_np = np.ones((1024, 1024)).astype(np.float32)
    s1 = mindspore.hal.Stream()
    for _ in range(10):
        x = Tensor(input_np)
        y = Tensor(input_np)

        with mindspore.hal.StreamCtx(s1):
            for _ in range(1000):
                out = mint.matmul(x, y)
            event = s1.record_event()

        cur_stream = mindspore.hal.current_stream()
        cur_stream.wait_event(event)

        out = func(out)
        assert np.allclose(out.asnumpy(), np.sin(np.matmul(input_np, input_np)))


def test_jit_within_multi_stream():
    """
    Feature: PyNative jit multi-stream.
    Description: Test running PyNative with stream/event.
    Expectation: run success
    """
    input_np = np.ones((1024, 1024)).astype(np.float32)
    s1 = mindspore.hal.Stream()
    for _ in range(10):
        x = Tensor(input_np)
        y = Tensor(input_np)

        @jit
        def matmul_jit(a, b):
            return mint.matmul(a, b)

        with mindspore.hal.StreamCtx(s1):
            for _ in range(1000):
                out = matmul_jit(x, y)
            event = s1.record_event()

        cur_stream = mindspore.hal.current_stream()
        cur_stream.wait_event(event)

        out = ops.sin(out)
        assert np.allclose(out.asnumpy(), np.sin(np.matmul(input_np, input_np)))


def test_multi_stream_with_jit_output():
    """
    Feature: PyNative jit multi-stream.
    Description: Test running PyNative with stream/event.
    Expectation: run success
    """
    input_np = np.ones((1024, 1024)).astype(np.float32)
    s1 = mindspore.hal.Stream()
    for _ in range(10):
        x = Tensor(input_np)
        y = Tensor(input_np)

        @jit
        def matmul_jit(a, b):
            return mint.matmul(a, b)

        for _ in range(1000):
            out = matmul_jit(x, y)

        cur_stream = mindspore.hal.current_stream()
        event = cur_stream.record_event()
        with mindspore.hal.StreamCtx(s1):
            s1.wait_event(event)
            out = ops.sin(out)

        assert np.allclose(out.asnumpy(), np.sin(np.matmul(input_np, input_np)))


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2136.03)
def test_pynative_and_graph_mixed_run():
    """
    Feature: test pynative and graph mixed run
    Description: single op run in pynative, the output to net input which run in graph
    Expectation: run success
    """
    test_pynative_with_jit()
    test_multi_stream_with_jit()
    test_multi_stream_with_event()
    test_jit_within_multi_stream()
    test_multi_stream_with_jit_output()
