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
"""Test with StreamCtx."""
# pylint: disable=W1514
# pylint: disable=W0621
import os
import re
import shutil
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, Parameter, nn
from mindspore.runtime import Stream
from mindspore.runtime import StreamCtx as MsJitStreamCtx
from mindspore.ops.functional import grad
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.GRAPH_MODE)


class MyMsJitStreamCtx(MsJitStreamCtx):
    def __init__(self, ctx_stream):
        self.stream = ctx_stream
        self.prev_stream = None

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


a = Tensor(np.ones([3, 3]), ms.float32)
b = Tensor(np.ones([3, 3]), ms.float32)


def clean_all_ir_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ir') or file_name.endswith('.dot') or \
                    file_name.endswith('.dat') or file_name.endswith('.pb'):
                os.remove(os.path.join(folder_path, file_name))


def find_newest_validateir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'\d+_validate_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def read_file(save_path):
    filename = find_newest_validateir_file(save_path)
    with open((os.path.join(filename)), 'r') as f:
        content = f.read()
    clean_all_ir_files(save_path)
    return content


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=6.00)
def test_my_ms_jit_stream_ctx():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()

    class MyMsJitStreamCtxNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with MyMsJitStreamCtx(s1):
                z = a + b + x
            return z - y

    save_path = "./test_my_ms_jit_stream_ctx"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert (out.asnumpy() == x.asnumpy()).all()
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 1

    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=8.00)
def test_my_ms_jit_stream_ctx_runtime():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()

    class MyMsJitStreamCtxNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with ms.runtime.StreamCtx(s1):
                z = a + b + x
            return z - y

    save_path = "./test_my_ms_jit_stream_ctx_runtime"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert (out.asnumpy() == x.asnumpy()).all()
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 1
    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=8.00)
def test_my_ms_jit_stream_ctx_multi():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()

    class MyMsJitStreamCtxMutliNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with MyMsJitStreamCtx(s1):
                z = a + b + x
            with MyMsJitStreamCtx(s2):
                y = a - y
            return y + z

    save_path = "./test_my_ms_jit_stream_ctx_multi"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxMutliNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert (out.asnumpy() == (x * 2).asnumpy()).all()
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 2
    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_my_ms_jit_stream_ctx_nest():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()

    class MyMsJitStreamCtxMutliNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with MyMsJitStreamCtx(s1):
                z = a + b + x
                with MyMsJitStreamCtx(s2):
                    y = a - y
            return y + z

    with pytest.raises(RuntimeError) as info:
        net = MyMsJitStreamCtxMutliNet()
        x = Tensor(np.ones([3, 3]), ms.float32)
        net(x)
    assert "Nested with StreamCtx statements are not supported in graph mode." in str(info.value)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=8.00)
def test_basic_stream_block_annotation_1():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()

    class MsJitStreamNet(nn.Cell):
        def construct(self, x, con):
            y = x * 2
            with MsJitStreamCtx(s1):
                z = a + x
                if con < 5:
                    z = z + 5
                y = y + z
            y = y + z
            return y + z

    save_path = "./test_basic_stream_block_annotation_1"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    net = MsJitStreamNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    con = 0
    result = net(x, con)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert np.allclose(result, Tensor(np.ones([3, 3], dtype=np.float32)) * 23)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 3
    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, con)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, con)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=12.00)
def test_basic_stream_block_annotation_2():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()

    class SimpleStreamNet(nn.Cell):
        def construct(self, x):
            x = ops.abs(x)
            with MsJitStreamCtx(s1):
                x = x + Tensor(np.ones([2, 2], dtype=np.float32))
                x = ops.relu(x)
            x = ops.abs(x)
            with MsJitStreamCtx(s2):
                x = x + Tensor(np.ones([2, 2], dtype=np.float32))
            x = ops.abs(x)
            return x

    save_path = "./test_basic_stream_block_annotation_2"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    x = Tensor(np.ones([2, 2], dtype=np.float32))
    net = SimpleStreamNet()
    result = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)) * 3)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 3
    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_nested_stream_blocks():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()

    class NestedStreamNet(nn.Cell):
        def construct(self, x):
            with MsJitStreamCtx(s1):
                x1 = x + Tensor(np.ones([2, 2], dtype=np.float32))
                with MsJitStreamCtx(s2):
                    x2 = x1 + Tensor(np.ones([2, 2], dtype=np.float32))
                    x3 = x2 + Tensor(np.ones([2, 2], dtype=np.float32))
                x4 = x3 - Tensor(np.ones([2, 2], dtype=np.float32))
            return x4

    with pytest.raises(RuntimeError) as info:
        x = Tensor(np.ones([2, 2], dtype=np.float32))
        net = NestedStreamNet()
        net(x)
    assert "Nested with StreamCtx statements are not supported in graph mode." in str(info.value)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_complex_nested_streams():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()
    s3 = Stream()
    class ComplexStreamsNet(nn.Cell):
        def construct(self, x):
            with MsJitStreamCtx(s1):
                t = Tensor(np.ones([2, 2], dtype=np.float32))
                x1 = x + t
                with MsJitStreamCtx(s2):
                    x2 = x1 + t
                    with MsJitStreamCtx(s3):
                        x3 = x2 + t
                    x4 = x3 + t
                x5 = x4 + t
            return x5

    with pytest.raises(RuntimeError) as info:
        x = Tensor(np.ones([2, 2], dtype=np.float32))
        net = ComplexStreamsNet()
        net(x)
    assert "Nested with StreamCtx statements are not supported in graph mode." in str(info.value)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=12.00)
def test_multiple_independent_streams():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()

    class MutliStreamsNet(nn.Cell):
        def construct(self, x, y):
            with MsJitStreamCtx(s1):
                x1 = x + Tensor(np.ones([2, 2], dtype=np.float32))
                x_relu = ops.relu(x1)

            intermediate = x + y

            with MsJitStreamCtx(s2):
                y1 = y + Tensor(np.ones([2, 2], dtype=np.float32))
                y_relu = ops.relu(y1)

            result = x_relu + y_relu
            result = result + intermediate
            return result

    save_path = "./test_multiple_independent_streams"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    x = Tensor(np.ones([2, 2], dtype=np.float32))
    y = Tensor(np.ones([2, 2], dtype=np.float32))
    net = MutliStreamsNet()
    result = net(x, y)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    no_inline_num = re.findall('no_inline', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)) * 6)
    assert len(stream_id_num) == 4
    assert len(no_inline_num) == 0
    ms.set_context(save_graphs=False)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, y)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, y)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=8.00)
def test_stream_with_control_flow():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()

    class ConditionalStreamNet(nn.Cell):
        def construct(self, x, use_stream):
            if use_stream:
                with MsJitStreamCtx(s1):
                    x = x + Tensor(np.ones([2, 2], dtype=np.float32))
            else:
                x = x + Tensor(np.ones([2, 2], dtype=np.float32))
            return x

    x = Tensor(np.ones([2, 2], dtype=np.float32))
    net = ConditionalStreamNet()
    result_with_stream = net(x, True)
    result_without_stream = net(x, False)
    assert np.allclose(result_with_stream, result_without_stream)
    assert np.allclose(result_with_stream, Tensor(np.ones([2, 2], dtype=np.float32)) * 2)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, True)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, False)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@pytest.mark.skip(reason='Not support yet, UnboundLocalError')
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_switch_like_control_flow_with_streams():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()
    s3 = Stream()
    class ConditionalStreamNet(nn.Cell):
        def construct(self, x, condition):
            result = x * 2
            t = Tensor(np.ones([2, 2], dtype=np.float32))
            if condition == 0:
                with MsJitStreamCtx(s1):
                    result = result + t
                    return result
            elif condition == 1:
                with MsJitStreamCtx(s2):
                    result = result + t * 3
                result = result + t
            elif condition == 2:
                result = result + t * 5
                return result
            else:
                result = result - t
                with MsJitStreamCtx(s3):
                    result = result + t * 7
            result = ops.relu(result)
            return result

    x = Tensor(np.ones([2, 2], dtype=np.float32))

    result0 = ConditionalStreamNet()(x, 0)
    assert np.allclose(result0, Tensor(np.ones([2, 2], dtype=np.float32)) * 3)

    result1 = ConditionalStreamNet()(x, 1)
    assert np.allclose(result1, Tensor(np.ones([2, 2], dtype=np.float32)) * 6)

    result2 = ConditionalStreamNet()(x, 2)
    assert np.allclose(result2, Tensor(np.ones([2, 2], dtype=np.float32)) * 7)

    result3 = ConditionalStreamNet()(x, 3)
    assert np.allclose(result3, Tensor(np.ones([2, 2], dtype=np.float32)) * 8)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=8.00)
def test_stream_with_break_statement():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()

    class StreamWithBreakNet(nn.Cell):
        def construct(self, x, max_iter):
            result = x
            i = 0
            t = Tensor(np.ones([2, 2], dtype=np.float32))
            while i < max_iter:
                with MsJitStreamCtx(s1):
                    result = result + t
                    if i >= 2:
                        result = result - t
                        break
                    result = result + t * 2
                i += 1
            return result

    x = Tensor(np.ones([2, 2], dtype=np.float32))
    net = StreamWithBreakNet()
    result = net(x, 5)
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)) * 7)
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, 5)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, 5)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=8.00)
def test_stream_with_continue_statement():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()

    class StreamWithContinueNet(nn.Cell):
        def construct(self, x, max_iter):
            result = x
            t = Tensor(np.ones([2, 2], dtype=np.float32))
            i = 0
            while i < max_iter:
                with MsJitStreamCtx(s1):
                    if i % 2 == 0:
                        result = result - t
                        i += 1
                        continue
                    result = result + t
                i += 1
            return result

    x = Tensor(np.ones([2, 2], dtype=np.float32))
    net = StreamWithContinueNet()
    result = net(x, 4)
    assert np.allclose(result, Tensor(np.ones([2, 2], dtype=np.float32)))
    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, 4)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, 4)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=10.00)
def test_stream_with_return_statement():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()

    class StreamWithReturnNet(nn.Cell):
        def construct(self, x, early_return):
            x = ops.abs(x)
            t = Tensor(np.ones([2, 2], dtype=np.float32))
            with MsJitStreamCtx(s1):
                x1 = x + t
                if early_return:
                    return x1
                relu1 = ops.relu(x1)
            with MsJitStreamCtx(s2):
                x2 = relu1 + t
                result = ops.relu(x2)
            return result

    x = Tensor(np.ones([2, 2], dtype=np.float32))
    net = StreamWithReturnNet()
    result_early = net(x, True)
    assert np.allclose(result_early, Tensor(np.ones([2, 2], dtype=np.float32)) * 2)

    result_full = net(x, False)
    assert np.allclose(result_full, Tensor(np.ones([2, 2], dtype=np.float32)) * 3)

    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, True)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, False)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=12.00)
def test_multiple_returns_in_different_streams():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()
    s3 = Stream()
    class StreamWithReturnNet(nn.Cell):
        def construct(self, x, condition):
            t = Tensor(np.ones([2, 2], dtype=np.float32))
            with MsJitStreamCtx(s1):
                x1 = x + t
                if condition == 1:
                    return x1
            with MsJitStreamCtx(s2):
                x2 = x + t * 2
                if condition == 2:
                    return x2
            with MsJitStreamCtx(s3):
                x3 = x1 + x2
                return x3

    x = Tensor(np.ones([2, 2], dtype=np.float32))
    net = StreamWithReturnNet()
    result1 = net(x, 1)
    assert np.allclose(result1, Tensor(np.ones([2, 2], dtype=np.float32)) * 2)

    result2 = net(x, 2)
    assert np.allclose(result2, Tensor(np.ones([2, 2], dtype=np.float32)) * 3)

    result3 = net(x, 3)
    assert np.allclose(result3, Tensor(np.ones([2, 2], dtype=np.float32)) * 5)

    ms.set_context(mode=ms.context.PYNATIVE_MODE)
    pynative_grad_out = grad(net)(x, 1)
    ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    graph_grad_out = grad(net)(x, 1)
    assert (pynative_grad_out.asnumpy() == graph_grad_out.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=6.00)
def test_with_stream_with_morph():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()

    def infer_dtype(args):
        return args

    def infer_shape(args):
        return args

    def mul_by(*args):
        def inner(x):
            with ms.runtime.StreamCtx(s1):
                y = args[0] * x
            return y - 1

        return inner

    class MorphNet(nn.Cell):
        def __init__(self):
            super().__init__()
            np_weight0 = np.array([1.0, 2.0, 3.0])
            np_weight1 = np.array([4.0, 5.0, 6.0])
            self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")
            self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")
            self.mul_by_100 = ops.Morph(mul_by(100), infer_shape, infer_dtype)

        def construct(self, x):
            input_a = x * self.weight0
            input_b = self.mul_by_100(input_a)
            out = input_b * self.weight1
            return out

    save_path = "./test_with_stream_with_morph"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    np_input_x = np.array([7.0, 8.0, 9.0])
    input_x = Tensor(np_input_x, ms.float32)
    net = MorphNet()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    grad_net(input_x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    ms.set_context(save_graphs=False)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=6.00)
def test_with_stream_with_morph_2():
    """
    Feature: Support with stream with morph.
    Description: Support with stream with morph.
    Expectation: Run success.
    """

    s1 = Stream()

    def infer_dtype(args):
        return args

    def infer_shape(args):
        return args

    def mul_by(*args):
        def inner(x):
            y = args[0] * x
            return y - 1
        return inner

    class MorphNet(nn.Cell):
        def __init__(self):
            super().__init__()
            np_weight0 = np.array([1.0, 2.0, 3.0])
            np_weight1 = np.array([4.0, 5.0, 6.0])
            self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")
            self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")
            self.mul_by_100 = ops.Morph(mul_by(100), infer_shape, infer_dtype)

        def construct(self, x):
            with ms.runtime.StreamCtx(s1):
                input_a = x * self.weight0
                input_b = self.mul_by_100(input_a)
            out = input_b * self.weight1
            return out

    save_path = "./test_with_stream_with_morph_2"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    np_input_x = np.array([7.0, 8.0, 9.0])
    input_x = Tensor(np_input_x, ms.float32)
    net = MorphNet()
    net(input_x)

    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    ms.set_context(save_graphs=False)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 6

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=6.00)
def test_my_ms_jit_stream_ctx_self():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """

    class MyMsJitStreamCtxNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.s5 = Stream()

        def construct(self, x):
            y = x * 2
            with MyMsJitStreamCtx(self.s5):
                z = a + b + x
            return z - y

    save_path = "./test_my_ms_jit_stream_ctx_self"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    assert (out.asnumpy() == x.asnumpy()).all()
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 1

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=6.00)
def test_my_ms_jit_stream_ctx_list():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s4 = Stream()
    s6 = Stream()
    stream_list = [s4, s6]
    class MyMsJitStreamCtxNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with MyMsJitStreamCtx(stream_list[0]):
                z = a + b + x
            return z - y

    save_path = "./test_my_ms_jit_stream_ctx_list"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert (out.asnumpy() == x.asnumpy()).all()
    assert len(stream_id_num) == 1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2352.00)
def test_with_stream_event_with_morph_cse():
    """
    Feature: Support with stream event in morph.
    Description: Support with stream event in morph.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()
    def infer_dtype(args):
        return args

    def infer_shape(args):
        return args

    def mul_by(*args):
        def inner(input_x):
            event = ms.runtime.Event()
            event = ops.Depend()(event, input_x)
            event = event.record()
            output = []
            with ms.runtime.StreamCtx(s1):
                event_end_recv = event.wait()
                x = ops.Depend()(input_x, event_end_recv)
                x = ops.matmul(x, x)
                output.append(x)
                event1 = ms.runtime.Event()
                event1 = ops.Depend()(event1, x)
                event1 = event1.record()

            with ms.runtime.StreamCtx(s2):
                event_end_recv = event.wait()
                x = ops.Depend()(input_x, event_end_recv)
                x = ops.matmul(x, x + 1)
                output.append(x)
                event2 = ms.runtime.Event()
                event2 = ops.Depend()(event2, x)
                event2 = event2.record()

            event_end_recv = event1.wait()
            output = ops.Depend()(output, event_end_recv)
            event_end_recv = event2.wait()
            output = ops.Depend()(output, event_end_recv)
            output = output[0] + output[1]
            return output * 0.0000000001

        return inner

    class MorphNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight0 = Parameter(ops.ones((8192, 8192), dtype=ms.float32), name="weight0")
            self.weight1 = Parameter(ops.ones((8192, 8192), dtype=ms.float32), name="weight1")
            self.mul_by_100 = ops.Morph(mul_by(100), infer_shape, infer_dtype)

        def construct(self, x):
            input_a = ops.matmul(x, self.weight0)
            input_a = input_a * 0.0
            input_a = ops.matmul(input_a, self.weight0)
            out = self.mul_by_100(input_a)
            out = out * self.weight1
            return out

    save_path = "./test_with_stream_event_with_morph_cse"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)

    input_x = ops.ones((8192, 8192), dtype=ms.float32)
    net = MorphNet()
    net(input_x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    content = read_file(save_path)
    stream_recv_num = re.findall('= StreamRecv', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_recv_num) == 4






@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=814.00)
def test_with_stream_with_morph_for_loop():
    """
    Feature: Support with stream in for loop.
    Description: Support with stream in for loop.
    Expectation: Run success.
    """
    loop_size = 2
    stream_list_2 = []
    for _ in range(loop_size):
        stream_list_2.append(Stream())
    def infer_dtype(args):
        return args

    def infer_shape(args):
        return args

    def mul_by(*args):
        def inner(input_x):
            for step_id in stream_list_2:
                with ms.runtime.StreamCtx(step_id):
                    input_x = ops.matmul(input_x, input_x)
            return input_x

        return inner

    class MorphNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul_by_100 = ops.Morph(mul_by(100), infer_shape, infer_dtype)

        def construct(self, x):
            out = self.mul_by_100(x)
            return out

    save_path = "./test_with_stream_with_morph_loop"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    input_x = ops.ones((8192, 8192), dtype=ms.float32)
    net = MorphNet()
    net(input_x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id: I64', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 2


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=814.00)
def test_with_stream_while_loop():
    """
    Feature: Support with stream in while loop.
    Description: Support with stream in while loop.
    Expectation: Run success.
    """
    loop_size = 2
    stream_list_2 = []
    for _ in range(loop_size):
        stream_list_2.append(Stream())
    class StreamWhileNet(nn.Cell):
        def construct(self, x):
            step_id = 0
            while step_id < len(stream_list_2):
                with ms.runtime.StreamCtx(stream_list_2[step_id]):
                    x = ops.matmul(x, x)
                step_id = step_id + 1
            return x

    save_path = "./test_with_stream_while_loop"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    input_x = ops.ones((8192, 8192), dtype=ms.float32)
    net = StreamWhileNet()
    net(input_x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id: I64', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 2
