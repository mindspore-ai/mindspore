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
"""Test with StreamCtx and event"""
import os
import re
import shutil
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor, ops
from mindspore.runtime import Stream, StreamCtx


ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={'jit_level': 'O0'})


def find_newest_ir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'.*_after_stream_assign_\d+_\d+\.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def read_file(save_path):
    filename = find_newest_ir_file(save_path)
    with open((os.path.join(filename)), 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def validate_stream_send_recv(content):
    stream_send_count = 0
    stream_recv_count = 0

    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue

        if 'primitive_attrs: {' in line:
            prim_attrs = line.split('primitive_attrs: {')[1].split('}')[0]
        else:
            continue

        if '= StreamSend' in line:
            has_stream_id = 'stream_id:' in prim_attrs
            has_event_id = 'event_id:' in prim_attrs
            has_record = 'record_event:' in prim_attrs
            if has_stream_id and has_event_id and has_record:
                stream_send_count += 1

        elif '= StreamRecv' in line:
            has_stream_id = 'stream_id:' in prim_attrs
            has_event_id = 'event_id:' in prim_attrs
            has_wait = 'wait_event:' in prim_attrs
            has_record_stream = 'record_event_stream:' in prim_attrs
            if has_stream_id and has_event_id and has_wait and has_record_stream:
                stream_recv_count += 1

    return stream_send_count, stream_recv_count


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


def test_single_withstream_multi_event():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """
    s1 = Stream()
    class EventNet(nn.Cell):
        def __init__(self):
            super().__init__()
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

    save_path = "./test_single_withstream_multi_event"
    os.environ['MS_DEV_SAVE_GRAPHS'] = '2'
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'after_stream_assign'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = save_path
    ms.set_context(jit_config={"jit_level": "O0"})

    x = Tensor(np.ones([3, 3]), ms.float32)
    net = EventNet()
    out = net(x)
    content = read_file(save_path)
    valid_streamsend_count, valid_streamrecv_count = validate_stream_send_recv(content)

    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass

    assert (out.asnumpy() == (-2 * x).asnumpy()).all()
    assert valid_streamsend_count == 2
    assert valid_streamrecv_count == 2


def test_multi_withstream_single_event():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()
    s3 = Stream()

    class EventNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.depend = ops.Depend()

        def construct(self, x):
            event = ms.runtime.Event()
            y = x * 2
            event = self.depend(event, y)
            event.record()
            with MyMsJitStreamCtx(s1):
                event.wait()
                y = self.depend(y, event)
                z1 = y + a
            with MyMsJitStreamCtx(s2):
                event.wait()
                y = self.depend(y, event)
                z2 = y + b
            with MyMsJitStreamCtx(s3):
                event.wait()
                y = self.depend(y, event)
                z3 = y - a + b
            return z1, z2, z3

    save_path = "./test_multi_withstream_single_event"
    os.environ['MS_DEV_SAVE_GRAPHS'] = '2'
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'after_stream_assign'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = save_path
    ms.set_context(jit_config={"jit_level": "O0"})

    x = Tensor(np.ones([3, 3]), ms.float32)
    net = EventNet()
    out1, out2, out3 = net(x)
    content = read_file(save_path)
    vali_streamsend_count, vali_streamrecv_count = validate_stream_send_recv(content)

    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass

    assert (out1.asnumpy() == (3 * x).asnumpy()).all()
    assert (out2.asnumpy() == (3 * x).asnumpy()).all()
    assert (out3.asnumpy() == (2 * x).asnumpy()).all()
    assert vali_streamsend_count == 1
    assert vali_streamrecv_count == 3


def test_multi_withstream_multi_event():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()

    class EventNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.depend = ops.Depend()

        def construct(self, x):
            y = x * 2
            event1 = ms.runtime.Event()
            event2 = ms.runtime.Event()
            with MyMsJitStreamCtx(s1):
                z = a + x
                event1 = self.depend(event1, z)
                event1.record()
            with MyMsJitStreamCtx(s2):
                z1 = x - b
                event1 = self.depend(event1, z1)
                event1.wait()
                z = self.depend(z, event1)
                z = z + z1
                event2 = self.depend(event2, z)
                event2.record()
            event1 = self.depend(event1, y)
            event1.wait()
            y = self.depend(y, event1)
            y = y + a + b
            event2.wait()
            z = self.depend(z, event2)
            return y + z

    save_path = "./test_multi_withstream_multi_event"
    os.environ['MS_DEV_SAVE_GRAPHS'] = '2'
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'after_stream_assign'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = save_path
    ms.set_context(jit_config={"jit_level": "O0"})

    x = Tensor(np.ones([3, 3]), ms.float32)
    net = EventNet()
    out = net(x)
    content = read_file(save_path)
    vali_streamsend_count, vali_streamrecv_count = validate_stream_send_recv(content)

    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass

    assert (out.asnumpy() == (6 * x).asnumpy()).all()
    assert vali_streamsend_count == 2
    assert vali_streamrecv_count == 3


def test_eventwait_before_eventrecord():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode.
    Expectation: Run success.
    """
    class EventNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.depend = ops.Depend()

        def construct(self, x):
            event = ms.runtime.Event()
            y = x * 2
            event = self.depend(event, y)
            event.wait()
            z = a + b + x
            event = self.depend(event, z)
            event.record()
            event.wait()
            return y + z

    save_path = "./test_eventwait_before_eventrecord"
    os.environ['MS_DEV_SAVE_GRAPHS'] = '2'
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'after_stream_assign'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = save_path
    ms.set_context(jit_config={"jit_level": "O0"})

    x = Tensor(np.ones([3, 3]), ms.float32)
    net = EventNet()
    out = net(x)
    content = read_file(save_path)
    vali_streamsend_count, vali_streamrecv_count = validate_stream_send_recv(content)

    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass

    assert (out.asnumpy() == (5 * x).asnumpy()).all()
    assert vali_streamsend_count == 1
    assert vali_streamrecv_count == 1
