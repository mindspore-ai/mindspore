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
"""Test StreamLimitCtx"""
# pylint: disable=W1514
import os
import pytest
import re
import shutil
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, nn, context
from mindspore.runtime import Stream, StreamCtx, StreamLimitCtx
from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE)

class MyMsJitStreamCtx(StreamCtx):
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=6.00)
def test_with_stream_limit():
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
                with StreamLimitCtx(s1, 8, 8):
                    y = y + ops.abs(a)
            return z - y

    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    save_path = "./test_with_stream_limit"
    context.set_context(save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    vector_num = re.findall('vector_num', content)
    cube_num = re.findall('cube_num', content)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert np.allclose(out.asnumpy(), Tensor(np.zeros([3, 3]), ms.float32).asnumpy(), 1e-3, 1e-3)
    assert len(stream_id_num) == 2
    assert len(vector_num) == 1
    assert len(cube_num) == 1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=6.00)
def test_with_stream_limit_runtime():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()

    class MyMsJitStreamCtxNet(nn.Cell):
        def construct(self, x):
            with ms.runtime.StreamCtx(s1):
                y = x * 2
                with ms.runtime.StreamLimitCtx(s1, 8, 8):
                    z = a + b + x
            return z - y


    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    save_path = "./test_with_stream_limit_runtime"
    context.set_context(save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    vector_num = re.findall('vector_num', content)
    cube_num = re.findall('cube_num', content)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert np.allclose(out.asnumpy(), x.asnumpy(), 1e-3, 1e-3)
    assert len(stream_id_num) == 2
    assert len(vector_num) == 1
    assert len(cube_num) == 1



@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_with_stream_limit_diff_stream():
    """
    Feature: Support with stream limit context.
    Description: Support with stream limit context.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()

    class MyMsJitStreamCtxNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with MyMsJitStreamCtx(s1):
                z = a + b + x
                with StreamLimitCtx(s2, 8, 8):
                    y = y + ops.abs(a)
            return z - y

    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    save_path = "./test_with_stream_limit_diff_stream"
    context.set_context(save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    vector_num = re.findall('vector_num', content)
    cube_num = re.findall('cube_num', content)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert np.allclose(out.asnumpy(), Tensor(np.zeros([3, 3]), ms.float32).asnumpy(), 1e-3, 1e-3)
    assert len(stream_id_num) == 2
    assert not vector_num
    assert not cube_num


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=518.00)
def test_with_stream_event_with_morph_multi():
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
                with ms.runtime.StreamLimitCtx(s1, 4, 8):
                    event_end_recv = event.wait()
                    x = ops.Depend()(input_x, event_end_recv)
                    output.append(x)
            with ms.runtime.StreamCtx(s2):
                with ms.runtime.StreamLimitCtx(s1, 5, 8):
                    event_end_recv = event.wait()
                    x = ops.Depend()(input_x, event_end_recv)
                    output.append(x)
            output = ops.Depend()(output, event_end_recv)

            return output

        return inner

    class MorphNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul_by_100 = ops.Morph(mul_by(100), infer_shape, infer_dtype)

        def construct(self, x):
            out = self.mul_by_100(x)
            return out

    save_path = "./test_with_stream_event_with_morph_multi"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    input_x = ops.ones((8192, 8192), dtype=ms.float32)
    net = MorphNet()
    net(input_x)

    ms.set_context(save_graphs=False)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    event_id_num = re.findall('event_id', content)
    vector_num = re.findall('vector_num', content)
    cube_num = re.findall('cube_num', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 6
    assert len(event_id_num) == 4
    assert len(vector_num) == 5
    assert len(cube_num) == 5


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=6.00)
def test_with_stream_limit_runtime_self():
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
            with ms.runtime.StreamCtx(self.s5):
                y = x * 2
                with ms.runtime.StreamLimitCtx(self.s5, 8, 8):
                    z = a + b + x
            return z - y


    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    save_path = "./test_with_stream_limit_runtime_self"
    context.set_context(save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    vector_num = re.findall('vector_num', content)
    cube_num = re.findall('cube_num', content)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert np.allclose(out.asnumpy(), x.asnumpy(), 1e-3, 1e-3)
    assert len(stream_id_num) == 2
    assert len(vector_num) == 1
    assert len(cube_num) == 1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=6.00)
def test_with_stream_limit_diff_stream_list():
    """
    Feature: Support with stream limit context.
    Description: Support with stream limit context.
    Expectation: Run success.
    """
    s4 = Stream()
    s6 = Stream()
    stream_list = [s4, s6]
    class MyMsJitStreamCtxNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with MyMsJitStreamCtx(stream_list[1]):
                z = a + b + x
                with StreamLimitCtx(stream_list[1], 8, 8):
                    y = y + ops.abs(a)
            return z - y

    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    save_path = "./test_with_stream_limit_diff_stream"
    context.set_context(save_graphs=True, save_graphs_path=save_path)
    net = MyMsJitStreamCtxNet()
    x = Tensor(np.ones([3, 3]), ms.float32)
    out = net(x)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    vector_num = re.findall('vector_num', content)
    cube_num = re.findall('cube_num', content)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert np.allclose(out.asnumpy(), Tensor(np.zeros([3, 3]), ms.float32).asnumpy(), 1e-3, 1e-3)
    assert len(stream_id_num) == 2
    assert len(vector_num) == 1
    assert len(cube_num) == 1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=776.00)
def test_with_stream_event_with_morph_multi_cse():
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
                with ms.runtime.StreamLimitCtx(s1, 4, 8):
                    event_end_recv = event.wait()
                    x = input_x * 2
                    x = ops.Depend()(x, event_end_recv)
                    output.append(x)
            with ms.runtime.StreamCtx(s2):
                with ms.runtime.StreamLimitCtx(s2, 5, 8):
                    event_end_recv = event.wait()
                    x = input_x * 2
                    x = ops.Depend()(x, event_end_recv)
                    output.append(x)
            output = ops.Depend()(output, event_end_recv)

            return output

        return inner

    class MorphNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul_by_100 = ops.Morph(mul_by(100), infer_shape, infer_dtype)

        def construct(self, x):
            out = self.mul_by_100(x)
            return out

    save_path = "./test_with_stream_event_with_morph_multi_cse"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    input_x = ops.ones((8192, 8192), dtype=ms.float32)
    net = MorphNet()
    net(input_x)

    ms.set_context(save_graphs=False)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    event_id_num = re.findall('event_id', content)
    vector_num = re.findall('vector_num', content)
    cube_num = re.findall('cube_num', content)
    muls_num = re.findall('PrimFunc_Muls', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(stream_id_num) == 8
    assert len(event_id_num) == 4
    assert len(vector_num) == 7
    assert len(cube_num) == 7
    assert len(muls_num) == 2


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_with_stream_limit_nested():
    """
    Feature: Support with stream.
    Description: Support with stream.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()

    class MyMsJitStreamCtxNet(nn.Cell):
        def construct(self, x):
            y = x * 2
            with MyMsJitStreamCtx(s1):
                z = a + b + x
                with StreamLimitCtx(s1, 8, 8):
                    y = y + ops.abs(a)
                    with StreamLimitCtx(s2, 2, 2):
                        y2 = y + ops.abs(a)
            return z - y2

    with pytest.raises(RuntimeError) as info:
        net = MyMsJitStreamCtxNet()
        x = Tensor(np.ones([3, 3]), ms.float32)
        net(x)
    assert "Nested with StreamLimitCtx statements are not supported in graph mode." in str(info.value)
