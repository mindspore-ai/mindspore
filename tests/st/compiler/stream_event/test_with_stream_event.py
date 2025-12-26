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
"""Test with StreamCtx"""
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=W1514
# pylint: disable=R1725
import os
import re
import shutil
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops, Parameter
from mindspore.runtime import Stream, StreamCtx
from mindspore.ops.functional import grad
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={'jit_level': 'O0'})


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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_stream_event():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """

    save_path = "./test_with_stream_event"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)

    s1 = Stream()
    class WithEventNet1(nn.Cell):
        def __init__(self):
            super(WithEventNet1, self).__init__()
            self.depend = ops.Depend()

        def construct(self, x):
            y = x * 2
            event = ms.runtime.Event()
            with MyMsJitStreamCtx(s1):
                z = a + b + x
                event = self.depend(event, z)
                event.record()
                event.wait()
            return y + z

    x = Tensor(np.ones([3, 3]), ms.float32)
    net = WithEventNet1()
    out = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    ms.set_context(save_graphs=False)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    event_id_num = re.findall('event_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert (out.asnumpy() == (x * 5).asnumpy()).all()
    assert len(stream_id_num) == 5
    assert len(event_id_num) == 3


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_stream_event_grad():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """
    s1 = Stream()
    class WithEventNet1(nn.Cell):
        def __init__(self):
            super(WithEventNet1, self).__init__()
            self.depend = ops.Depend()

        def construct(self, x):
            y = x * 2
            event = ms.runtime.Event()
            with MyMsJitStreamCtx(s1):
                z = a + b + x
                event = self.depend(event, z)
                event.record()
                event.wait()
            return y + z

    ms.set_context(mode=ms.GRAPH_MODE)
    x = Tensor(np.ones([3, 3]), ms.float32)
    net = WithEventNet1()
    graph_grad_out = grad(net)(x)
    assert (graph_grad_out.asnumpy() == (3 * x).asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_event_multi_with_streams():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()
    class WithEventNet(nn.Cell):
        def __init__(self):
            super(WithEventNet, self).__init__()
            self.depend = ops.Depend()

        def construct(self, x):
            y = x * 2
            event = ms.runtime.Event()
            with MyMsJitStreamCtx(s1):
                z = a + b + x
                event = self.depend(event, z)
                event.record()
            with MyMsJitStreamCtx(s2):
                z1 = a + b + x
                event = self.depend(event, z1)
                event.wait()
                z = z + z1
            y = self.depend(y, z)
            return y + z

    save_path = "./test_event_multi_with_streams"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)

    x = Tensor(np.ones([3, 3]), ms.float32)
    net = WithEventNet()
    out = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    ms.set_context(save_graphs=False)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    event_id_num = re.findall('event_id', content)
    add_num = re.findall('PrimFunc_Add', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert (out.asnumpy() == (x * 8).asnumpy()).all()
    assert len(stream_id_num) == 10
    assert len(event_id_num) == 4
    assert len(add_num) == 4


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_stream_multi_events():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """
    s1 = Stream()

    class WithEventNet(nn.Cell):
        def __init__(self):
            super(WithEventNet, self).__init__()
            self.depend = ops.Depend()

        def construct(self, x):
            y = x * 2
            event1 = ms.runtime.Event()
            event2 = ms.runtime.Event()
            with MyMsJitStreamCtx(s1):
                event1.record()
                event1.wait()
                output = y + x
                event2 = self.depend(event2, output)
                event2.record()
            event2 = self.depend(event2, y)
            event2.wait()
            output = self.depend(output, event2)
            output = x - y * (output / 2)
            return output

    save_path = "./test_with_stream_multi_events"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    x = Tensor(np.ones([3, 3]), ms.float32)
    net = WithEventNet()
    out = net(x)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    ms.set_context(save_graphs=False)
    content = read_file(save_path)
    event_id_num = re.findall('event_id', content)
    stream_id_num = re.findall('stream_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert (out.asnumpy() == (-2 * x).asnumpy()).all()
    assert len(event_id_num) == 6
    assert len(stream_id_num) == 8


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_stream_event_with_morph():
    """
    Feature: Support with stream event in morph.
    Description: Support with stream event in morph.
    Expectation: Run success.
    """
    s1 = Stream()

    def infer_dtype(args):
        return args

    def infer_shape(args):
        return args

    def mul_by(*args):
        def inner(x):
            event = ms.runtime.Event()
            with ms.runtime.StreamCtx(s1):
                y = args[0] * x
                event = ops.Depend()(event, y)
                event.record()
                event.wait()
            return y - 1

        return inner

    class MorphNet(nn.Cell):
        def __init__(self):
            super(MorphNet, self).__init__()
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

    save_path = "./test_with_stream_event_with_morph"
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
    event_id_num = re.findall('event_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(event_id_num) == 3


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_stream_event_list():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """
    s1 = Stream()

    class WithEventNet2(nn.Cell):
        def __init__(self):
            super().__init__()
            self.depend = ops.Depend()

        def construct(self, x, rank_num):
            y = x * 2
            event_list = []
            z = x + 1
            for _ in range(rank_num):
                event = ms.runtime.Event()
                with MyMsJitStreamCtx(s1):
                    z = a + b + x
                    event = self.depend(event, z)
                    event.record()
                    event_list.append(event)
                    event.wait()
            for event_index in event_list:
                event_end_recv = event_index.wait()
                y = self.depend(y, event_end_recv)
            return y + z

    save_path = "./test_with_stream_event_list"
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    x = Tensor(np.ones([3, 3]), ms.float32)
    net = WithEventNet2()
    out = net(x, 2)
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    ms.set_context(save_graphs=False)
    content = read_file(save_path)
    stream_id_num = re.findall('stream_id', content)
    event_id_num = re.findall('event_id', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert (out.asnumpy() == (x * 5).asnumpy()).all()
    assert len(stream_id_num) == 9
    assert len(event_id_num) == 8


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cust_class_record_wait():
    """
    Feature: getattr for custom class.
    Description: Support getattr for custom class.
    Expectation: No exception.
    """

    class GetattrClass():
        def __init__(self):
            self.attr1 = 99
            self.attr2 = 1

        def record(self):
            return self.attr1

        def wait(self):
            self.attr1 = 10
            return self.attr2

    class GetattrClassNet(ms.nn.Cell):
        def __init__(self):
            super(GetattrClassNet, self).__init__()
            self.cls = GetattrClass()

        def construct(self):
            self.cls.record()
            self.cls.wait()
            return self.cls.attr1

    ms.set_context(mode=ms.GRAPH_MODE)
    net = GetattrClassNet()
    out = net()
    assert out == 10


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cust_class_record_wait_2():
    """
    Feature: getattr for custom class.
    Description: Support getattr for custom class.
    Expectation: No exception.
    """

    class GetattrClass():
        def __init__(self):
            self.attr1 = 99
            self.attr2 = 1

        def method1(self, x):
            return x + self.attr2 * 2

        def record(self):
            return self.attr1

        def wait(self):
            return self.attr2

    class GetattrClassNet(ms.nn.Cell):
        def __init__(self):
            super(GetattrClassNet, self).__init__()
            self.cls = GetattrClass()

        def construct(self):
            obj = self.cls
            obj.record()
            obj.wait()
            return self.cls.method1(self.cls.attr1)

    ms.set_context(mode=ms.GRAPH_MODE)
    net = GetattrClassNet()
    out = net()
    assert out == 101
