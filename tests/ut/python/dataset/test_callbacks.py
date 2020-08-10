# Copyright 2020 Huawei Technologies Co., Ltd
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
# ==============================================================================
from builtins import range, super
import time

import pytest

from mindspore import context
from mindspore import log as logger
from mindspore.dataset.callback import DSCallback, WaitedDSCallback
from mindspore.train import Model
from mindspore.train.callback import Callback

import mindspore.dataset as ds
import mindspore.nn as nn

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class MyDSCallback(DSCallback):
    def __init__(self, step_size=1, events=None, cb_id=0):
        super().__init__(step_size)
        self.events = events
        self.cb_id = cb_id

    def append(self, event_name, ds_run_context):
        event = [event_name, ds_run_context.cur_epoch_num,
                 ds_run_context.cur_step_num_in_epoch, ds_run_context.cur_step_num]
        event = '_'.join([str(e) for e in event])
        index = -1
        for i, e in enumerate(self.events):
            if e[0] == event:
                index = i
                break
        if index != -1:
            self.events[index][1].append(self.cb_id)
        else:
            self.events.append((event, [self.cb_id]))

    def ds_begin(self, ds_run_context):
        self.append("begin", ds_run_context)

    def ds_end(self, ds_run_context):
        self.append("end", ds_run_context)

    def ds_epoch_begin(self, ds_run_context):
        self.append("epoch_begin", ds_run_context)

    def ds_epoch_end(self, ds_run_context):
        self.append("epoch_end", ds_run_context)

    def ds_step_begin(self, ds_run_context):
        self.append("step_begin", ds_run_context)

    def ds_step_end(self, ds_run_context):
        self.append("step_end", ds_run_context)


def generate_expected(epoch_num, step_num, step_size=1, map_num=1, repeat=1):
    events = []
    cb_id = list(range(map_num))

    def append(name, e, s):
        event = [name, e + 1, s + 1, e * step_num * repeat + s + 1]
        event = '_'.join([str(ev) for ev in event])
        events.append((event, cb_id))

    events.append(("begin_0_0_0", cb_id))
    for e in range(epoch_num):
        append("epoch_begin", e, -1)
        for s in range(step_num * repeat):
            if s % step_size == 0:
                append("step_begin", e, s)
                append("step_end", e, s)
        append("epoch_end", e, step_num * repeat - 1)
    return events


def build_test_case_1cb(epochs, steps, step_size=1, repeat=1):
    events = []

    arr = list(range(1, steps + 1))
    data = ds.NumpySlicesDataset(arr, shuffle=False)

    my_cb = MyDSCallback(step_size=step_size, events=events)

    data = data.map(operations=(lambda x: x), callbacks=my_cb)
    if repeat != 1:
        data = data.repeat(repeat)
    itr = data.create_tuple_iterator(num_epochs=epochs)
    for _ in range(epochs):
        for _ in itr:
            pass

    expected_events = generate_expected(epochs, steps, step_size, 1, repeat)
    assert expected_events == events


def build_test_case_2cbs(epochs, steps):
    events1 = []
    events2 = []
    my_cb1 = MyDSCallback(events=events1)
    my_cb2 = MyDSCallback(events=events2)

    arr = list(range(1, steps + 1))
    data = ds.NumpySlicesDataset(arr, shuffle=False)

    data = data.map(operations=(lambda x: x), callbacks=[my_cb1, my_cb2])

    itr = data.create_tuple_iterator(num_epochs=epochs)
    for _ in range(epochs):
        for _ in itr:
            pass

    expected_events = generate_expected(epochs, steps)
    assert expected_events == events1
    assert expected_events == events2


def build_test_case_2maps(epochs, steps):
    events = []
    my_cb1 = MyDSCallback(events=events, cb_id=0)
    my_cb2 = MyDSCallback(events=events, cb_id=1)

    arr = list(range(1, steps + 1))
    data = ds.NumpySlicesDataset(arr, shuffle=False)

    data = data.map(operations=(lambda x: x), callbacks=my_cb1)
    data = data.map(operations=(lambda x: x), callbacks=my_cb2)

    itr = data.create_tuple_iterator(num_epochs=epochs)
    for _ in range(epochs):
        for _ in itr:
            pass

    expected_events = generate_expected(epochs, steps, map_num=2)

    assert expected_events[1:] == events[1:]

    for event in events:
        assert len(event) == 2
        event, cb_ids = event
        if event != "begin_0_0_0":
            assert cb_ids[0] == 0
            assert cb_ids[1] == 1


def test_callbacks_all_methods():
    logger.info("test_callbacks_all_methods")

    build_test_case_1cb(1, 1)
    build_test_case_1cb(1, 2)
    build_test_case_1cb(1, 3)
    build_test_case_1cb(1, 4)

    build_test_case_1cb(2, 1)
    build_test_case_1cb(2, 2)
    build_test_case_1cb(2, 3)
    build_test_case_1cb(2, 4)

    build_test_case_1cb(3, 1)
    build_test_case_1cb(3, 2)
    build_test_case_1cb(3, 3)
    build_test_case_1cb(3, 4)


def test_callbacks_var_step_size():
    logger.info("test_callbacks_var_step_size")

    build_test_case_1cb(1, 2, 2)
    build_test_case_1cb(1, 3, 2)
    build_test_case_1cb(1, 4, 2)

    build_test_case_1cb(2, 2, 2)
    build_test_case_1cb(2, 3, 2)
    build_test_case_1cb(2, 4, 2)

    build_test_case_1cb(3, 2, 2)
    build_test_case_1cb(3, 3, 2)
    build_test_case_1cb(3, 4, 2)


def test_callbacks_all_2cbs():
    logger.info("test_callbacks_all_2cbs")

    build_test_case_2cbs(4, 1)
    build_test_case_2cbs(4, 2)
    build_test_case_2cbs(4, 3)
    build_test_case_2cbs(4, 4)


def test_callbacks_2maps():
    logger.info("test_callbacks_2maps")

    build_test_case_2maps(5, 10)

    build_test_case_2maps(6, 9)


class MyWaitedCallback(WaitedDSCallback):
    def __init__(self, events, step_size=1):
        super().__init__(step_size)
        self.events = events

    def sync_epoch_begin(self, train_run_context, ds_run_context):
        event = f"ds_epoch_begin_{ds_run_context.cur_epoch_num}_{ds_run_context.cur_step_num}"
        self.events.append(event)

    def sync_step_begin(self, train_run_context, ds_run_context):
        event = f"ds_step_begin_{ds_run_context.cur_epoch_num}_{ds_run_context.cur_step_num}"
        self.events.append(event)


class MyMSCallback(Callback):
    def __init__(self, events):
        self.events = events

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        event = f"ms_epoch_end_{cb_params.cur_epoch_num}_{cb_params.cur_step_num}"
        self.events.append(event)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        event = f"ms_step_end_{cb_params.cur_epoch_num}_{cb_params.cur_step_num}"
        self.events.append(event)


class Net(nn.Cell):
    def construct(self, x, y):
        return x


def test_train_non_sink():
    logger.info("test_train_non_sink")

    events = []
    my_cb1 = MyWaitedCallback(events, 1)
    my_cb2 = MyMSCallback(events)
    arr = [1, 2, 3, 4]
    data = ds.NumpySlicesDataset((arr, arr), column_names=["c1", "c2"], shuffle=False)
    data = data.map(operations=(lambda x: x), callbacks=my_cb1)

    net = Net()
    model = Model(net)

    model.train(2, data, dataset_sink_mode=False, callbacks=[my_cb2, my_cb1])
    expected_synced_events = ['ms_step_end_1_1', 'ds_step_begin_1_2', 'ms_step_end_1_2', 'ds_step_begin_1_3',
                              'ms_step_end_1_3', 'ds_step_begin_1_4', 'ms_step_end_1_4',
                              'ms_epoch_end_1_4', 'ds_epoch_begin_2_4',
                              'ds_step_begin_2_5', 'ms_step_end_2_5', 'ds_step_begin_2_6',
                              'ms_step_end_2_6', 'ds_step_begin_2_7', 'ms_step_end_2_7', 'ds_step_begin_2_8',
                              'ms_step_end_2_8', 'ms_epoch_end_2_8']

    assert events == expected_synced_events


def test_train_batch_size2():
    logger.info("test_train_batch_size2")

    events = []
    my_cb1 = MyWaitedCallback(events, 2)
    my_cb2 = MyMSCallback(events)
    arr = [1, 2, 3, 4]
    data = ds.NumpySlicesDataset((arr, arr), column_names=["c1", "c2"], shuffle=False)
    data = data.map(operations=(lambda x: x), callbacks=my_cb1)
    data = data.batch(2)
    net = Net()
    model = Model(net)

    model.train(2, data, dataset_sink_mode=False, callbacks=[my_cb2, my_cb1])

    expected_synced_events = ['ms_step_end_1_1', 'ds_step_begin_1_3',
                              'ms_step_end_1_2',
                              'ms_epoch_end_1_2', 'ds_epoch_begin_2_4',
                              'ds_step_begin_2_5', 'ms_step_end_2_3', 'ds_step_begin_2_7',
                              'ms_step_end_2_4', 'ms_epoch_end_2_4']

    assert events == expected_synced_events


def test_callbacks_validations():
    logger.info("test_callbacks_validations")

    with pytest.raises(Exception) as err:
        data = ds.NumpySlicesDataset([1, 2, 3, 4], shuffle=False)
        data.map(operations=(lambda x: x), callbacks=0)
    assert "Argument callbacks with value 0 is not " in str(err.value)

    with pytest.raises(Exception) as err:
        my_cb1 = MyDSCallback()
        data = ds.NumpySlicesDataset([1, 2, 3, 4], shuffle=False)
        data.map(operations=(lambda x: x), callbacks=[my_cb1, 0])
    assert "Argument callbacks[1] with value 0 is not " in str(err.value)

    with pytest.raises(Exception) as err:
        class BadCB(DSCallback):
            pass

        my_cb = BadCB()

        data = ds.NumpySlicesDataset([1, 2, 3, 4], shuffle=False)
        data = data.map(operations=(lambda x: x), callbacks=my_cb)
        for _ in data:
            pass
    assert "Provided Callback class did not override any of the 6 callback methods." in str(err.value)


def test_callback_sink_simulation():
    logger.info("test_callback_sink_simulation")

    events = []
    epochs = 2
    my_cb = MyWaitedCallback(events, 1)
    data = ds.NumpySlicesDataset([1, 2, 3, 4], shuffle=False)
    data = data.map(operations=(lambda x: x), callbacks=my_cb)
    data = data.to_device()
    data.send(num_epochs=epochs)
    for e in range(epochs):
        for s in range(4):
            time.sleep(0.5)
            events.append(f"ms_step_end_{e + 1}_{e * 4 + s + 1}")
            my_cb.step_end(run_context=0)
        events.append(f"ms_epoch_end_{e + 1}_{(e + 1) * 4}")
        my_cb.epoch_end(run_context=0)
    expected_synced_events = ['ms_step_end_1_1', 'ds_step_begin_1_2', 'ms_step_end_1_2', 'ds_step_begin_1_3',
                              'ms_step_end_1_3', 'ds_step_begin_1_4', 'ms_step_end_1_4',
                              'ms_epoch_end_1_4', 'ds_epoch_begin_2_4',
                              'ds_step_begin_2_5', 'ms_step_end_2_5', 'ds_step_begin_2_6',
                              'ms_step_end_2_6', 'ds_step_begin_2_7', 'ms_step_end_2_7', 'ds_step_begin_2_8',
                              'ms_step_end_2_8', 'ms_epoch_end_2_8']

    assert events == expected_synced_events


def test_callbacks_repeat():
    logger.info("test_callbacks_repeat")

    build_test_case_1cb(epochs=2, steps=2, step_size=1, repeat=2)
    build_test_case_1cb(epochs=2, steps=2, step_size=1, repeat=3)
    build_test_case_1cb(epochs=2, steps=2, step_size=2, repeat=3)
    build_test_case_1cb(epochs=3, steps=2, step_size=4, repeat=3)


if __name__ == '__main__':
    test_callbacks_all_methods()
    test_callbacks_all_2cbs()
    test_callbacks_2maps()
    test_callbacks_validations()
    test_callbacks_var_step_size()
    test_train_batch_size2()
    test_callback_sink_simulation()
    test_callbacks_repeat()
