# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.context as context
from mindspore import Tensor, ops, grad, jit
import mindspore as ms
import numpy as np
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_simple_stream():
    """
    Feature: Hal stream api.
    Description: Test hal.Stream api.
    Expectation: hal.Stream api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    a = Tensor(2.0)
    s1 = ms.hal.Stream()
    with ms.hal.StreamCtx(s1):
        ops.abs(a)
    s1.synchronize()


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_set_stream():
    """
    Feature: Hal stream api.
    Description: Test hal.set_cur_stream api.
    Expectation: hal.set_cur_stream api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    cur_stream = ms.hal.current_stream()
    assert cur_stream == ms.hal.default_stream()
    s1 = ms.hal.Stream()
    ms.hal.set_cur_stream(s1)
    assert ms.hal.current_stream() == s1


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_stream_query():
    """
    Feature: Hal stream api.
    Description: Test hal.Stream.query api.
    Expectation: hal.Stream.query api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    a = Tensor(np.ones([5000, 5000]), ms.float32)
    s1 = ms.hal.Stream()
    with ms.hal.StreamCtx(s1):
        ops.bmm(a, a)
        assert not s1.query()

    s1.synchronize()
    assert s1.query()

@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_wait_stream():
    """
    Feature: Hal stream api.
    Description: Test hal.Stream.wait_stream api.
    Expectation: hal.Stream.wait_stream api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    s1 = ms.hal.Stream()
    s2 = ms.hal.Stream()

    a = Tensor(np.random.randn(20, 20), ms.float32)
    with ms.hal.StreamCtx(s1):
        b = ops.matmul(a, a)

    with ms.hal.StreamCtx(s2):
        s2.wait_stream(s1)
        c = ops.matmul(b, b)
    ms.hal.synchronize()
    assert np.allclose(ops.mm(a, a).numpy(), b.numpy())
    assert np.allclose(ops.mm(b, b).numpy(), c.numpy())


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_wait_event():
    """
    Feature: Hal stream api.
    Description: Test hal.Stream.wait_event api.
    Expectation: hal.Stream.wait_event api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    s1 = ms.hal.Stream()
    s2 = ms.hal.Stream()
    ev = ms.hal.Event()

    a = Tensor(np.random.randn(20, 20), ms.float32)
    with ms.hal.StreamCtx(s1):
        b = ops.matmul(a, a)
        ev.record(s1)
        assert ev.query() is False

    with ms.hal.StreamCtx(s2):
        s2.wait_event(ev)
        c = ops.matmul(b, b)
    ms.hal.synchronize()
    assert np.allclose(ops.mm(a, a).asnumpy(), b.asnumpy())
    assert np.allclose(ops.mm(b, b).asnumpy(), c.asnumpy())


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_synchronize():
    """
    Feature: Hal stream api.
    Description: Test hal.synchronize api.
    Expectation: hal.synchronize api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    s1 = ms.hal.Stream()
    s2 = ms.hal.Stream()

    a = Tensor(np.ones([1, 2]), ms.float32)
    b = Tensor(np.ones([2,]), ms.float32)
    with ms.hal.StreamCtx(s1):
        ops.matmul(a, b)

    with ms.hal.StreamCtx(s2):
        ops.matmul(a, b)
    ms.hal.synchronize()


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_jit_stream():
    """
    Feature: Hal stream api.
    Description: Test hal.StreamCtx api.
    Expectation: hal.StreamCtx api performs as expected in jit.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    s1 = ms.hal.Stream()
    event = ms.hal.Event()

    a = Tensor(np.random.randn(20, 20), ms.float32)
    b = Tensor(np.random.randn(20, 20), ms.float32)
    a *= 4
    event.record()
    with ms.hal.StreamCtx(s1):
        s1.wait_event(event)
        @jit
        def jit_func(a):
            return a + 2
        c = jit_func(a)
        d = ops.matmul(c, b)
    ms.hal.synchronize()

    assert np.allclose(d.asnumpy(), ops.mm((a + 2), b).asnumpy())


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_grad_stream():
    """
    Feature: Hal stream api.
    Description: Test hal.StreamCtx api.
    Expectation: hal.StreamCtx api performs as expected in grad.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f(x):
        return ops.sin(x)
    grad_fn = grad(f)

    a = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), ms.float32)
    s1 = ms.hal.Stream()
    with ms.hal.StreamCtx(s1):
        grad_a = grad_fn(a)
    s1.synchronize()
    assert np.allclose(grad_fn(a).asnumpy(), grad_a.asnumpy())


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_get_stream():
    """
    Feature: Hal stream api.
    Description: Test hal.cur_stream api.
    Expectation: hal.cur_stream api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    s1 = ms.hal.current_stream()
    s1.record_event()

    s1 = ms.hal.default_stream()
    s1.record_event()


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_multi_streams():
    """
    Feature: Hal stream api.
    Description: Test multi streams.
    Expectation: hal api performs as expected in multi streams.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    prev_curr_stream = ms.hal.current_stream()

    s1 = ms.hal.Stream()
    s2 = ms.hal.Stream()

    a = Tensor(np.random.randn(20, 20), ms.float32)
    with ms.hal.StreamCtx(s1):
        is_stream_stream1 = (ms.hal.current_stream() == s1)
        b = ops.matmul(a, a)
    ev = s1.record_event()
    is_stream_prev_curr_stream_1 = (ms.hal.current_stream() == prev_curr_stream)
    s2.wait_event(ev)

    with ms.hal.StreamCtx(s2):
        is_stream_stream2 = (ms.hal.current_stream() == s2)
        c = ops.matmul(b, b)
    s2.synchronize()
    is_stream_prev_curr_stream_2 = (ms.hal.current_stream() == prev_curr_stream)

    assert is_stream_stream1 is True
    assert is_stream_prev_curr_stream_1 is True
    assert is_stream_stream2 is True
    assert is_stream_prev_curr_stream_2 is True
    assert np.allclose(ops.matmul(a, a).asnumpy(), b.asnumpy())
    assert np.allclose(ops.matmul(b, b).asnumpy(), c.asnumpy())


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_hal_none_streams():
    """
    Feature: Hal stream api.
    Description: Test none streams.
    Expectation: hal api performs as expected in none streams.
    """
    curr_stream = ms.hal.current_stream()
    with ms.hal.StreamCtx(None):
        is_curr_stream_same = (ms.hal.current_stream() == curr_stream)

    assert is_curr_stream_same is True


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_hal_record_stream():
    """
    Feature: Hal record stream.
    Description: Test record stream.
    Expectation: hal api performs as expected.
    """
    s1 = ms.hal.Stream()

    one_input = Tensor(np.ones([16, 256, 256, 256]).astype(np.float32), ms.float32)
    weight = Tensor(np.ones([256, 256, 16, 16]).astype(np.float32), ms.float32)
    zero_input = Tensor(np.zeros([16, 256, 256, 256]).astype(np.float32), ms.float32)
    for _ in range(3):
        add_one = one_input + 1
        with ms.hal.StreamCtx(s1):
            # Use conv2d because this operator takes a long time in device.
            conv_res = ops.conv2d(add_one, weight)
            add_two = add_one + 1
        del add_one

        # Mess up the mem of add_one. The value of add_two will be wrong, if record_stream is not used.
        f = zero_input + 1

        # Recalculate results for compare.
        add_one1 = one_input + 1
        conv_res1 = ops.conv2d(add_one1, weight)
        add_two1 = add_one1 + 1
        assert np.allclose(add_two.asnumpy(), add_two1.asnumpy(), rtol=1e-3, atol=1e-03)
        del f, conv_res, conv_res1
