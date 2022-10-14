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
# ============================================================================
""" test_training """
import logging
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Model, context
from mindspore import Tensor
from mindspore.train.callback import Callback
from mindspore.train.callback import BackupAndRestore
from mindspore.nn.optim import Momentum
from ..ut_filter import non_graph_engine
from ....dataset_mock import MindData


class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal', pad_mode='valid')
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 222 * 222, 3)  # padding=0

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out


class LossNet(nn.Cell):
    """ LossNet definition """

    def __init__(self):
        super(LossNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal', pad_mode='valid')
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 222 * 222, 3)  # padding=0
        self.loss = nn.SoftmaxCrossEntropyWithLogits()

    def construct(self, x, y):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        out = self.loss(x, y)
        return out


class NetNoLoss(nn.Cell):
    def __init__(self, in_features, out_features):
        super(NetNoLoss, self).__init__()
        self.dense = nn.Dense(in_features, out_features)

    def construct(self, input_x):
        return self.dense(input_x)


def get_model(metrics=None):
    """ get_model """
    net = Net()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optim = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(net, loss_fn=loss, optimizer=optim, metrics=metrics)
    return model


def get_dataset():
    """ get_dataset """
    dataset_types = (np.float32, np.float32)
    dataset_shapes = ((32, 3, 224, 224), (32, 3))

    dataset = MindData(size=2, batch_size=32,
                       np_types=dataset_types,
                       output_shapes=dataset_shapes,
                       input_indexs=(0, 1))
    return dataset


class MindDataSet(MindData):
    def __init__(self, dataset_types, dataset_shapes):
        super(MindDataSet, self).__init__(size=2, batch_size=32,
                                          np_types=dataset_types,
                                          output_shapes=dataset_shapes,
                                          input_indexs=(0, 1))

    def __next__(self):
        if self._size < self._iter_num:
            raise StopIteration
        self._iter_num += 1
        lst = []
        for shape_, type_ in zip(self._output_shapes, self._np_types):
            lst.append(Tensor(np.ones(shape_).astype(type_)))
        return tuple(lst)


@non_graph_engine
def test_single_input():
    """ test_single_input """
    input_data = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32))
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(Net())
    out = model.predict(input_data)
    assert out is not None


@non_graph_engine
def test_multiple_argument():
    """ test_multiple_argument """
    input_data = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]).astype(np.float32))
    input_label = Tensor(np.random.randint(0, 3, [1, 3]).astype(np.float32))
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(LossNet())
    out = model.predict(input_data, input_label)
    assert out is not None


def test_train_feed_mode(test_with_simu):
    """ test_train_feed_mode """
    context.set_context(mode=context.GRAPH_MODE)
    dataset = get_dataset()
    model = get_model()
    if test_with_simu:
        return
    model.train(2, dataset)


def test_dataset_sink_mode_args_check():
    """ test_dataset_sink_mode_args_check """
    dataset = get_dataset()
    model = get_model()
    with pytest.raises(TypeError):
        model.train(2, dataset, dataset_sink_mode="True")

    with pytest.raises(TypeError):
        model.train(2, dataset, dataset_sink_mode=1)


def test_model_train_initial_epoch_error_param():
    """
    Feature: Model train
    Description: Train network with initial_epoch.
    Expectation: Raise error for initial_epoch.
    """
    dataset = get_dataset()
    model = get_model()
    with pytest.raises(TypeError):
        model.train(3, dataset, initial_epoch="123")

    with pytest.raises(ValueError):
        model.train(3, dataset, initial_epoch=-1)

    with pytest.raises(ValueError):
        model.train(3, dataset, initial_epoch=4)


class InitialEpoch(Callback):
    """ CallbackTest definition """
    def epoch_end(self, run_context):
        # only used to check cur_epoch_num
        cb_params = run_context.original_args()
        assert cb_params.cur_epoch_num == 2


def test_model_train_initial_epoch():
    """
    Feature: Model train
    Description: Train network with initial_epoch.
    Expectation: Raise error for initial_epoch.
    """
    context.set_context(mode=context.GRAPH_MODE)
    dataset_types = (np.float32, np.float32)
    dataset_shapes = ((16, 16), (16, 16))
    dataset = MindDataSet(dataset_types, dataset_shapes)
    net = NetNoLoss(16, 16)
    loss = nn.MSELoss()
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={"acc"}, amp_level="O0")
    model.train(2, dataset, dataset_sink_mode=False)
    model.train(2, dataset, dataset_sink_mode=False, initial_epoch=0)
    model.train(2, dataset, dataset_sink_mode=False, initial_epoch=1)
    model.train(2, dataset, dataset_sink_mode=True, initial_epoch=1)
    initial_epoch = InitialEpoch()
    model.train(2, dataset, callbacks=initial_epoch, dataset_sink_mode=True, initial_epoch=1)
    model.train(2, dataset, callbacks=initial_epoch, dataset_sink_mode=False, initial_epoch=1)


def test_model_callback_restore():
    """
    Feature: Model train
    Description: Train network with restore callback.
    Expectation: Exec success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    dataset_types = (np.float32, np.float32)
    dataset_shapes = ((16, 16), (16, 16))
    dataset = MindDataSet(dataset_types, dataset_shapes)
    net = NetNoLoss(16, 16)
    loss = nn.MSELoss()
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={"acc"}, amp_level="O0")
    backup_cb = BackupAndRestore("backup", save_freq="epoch", delete_checkpoint=True)
    # backup
    model.train(3, dataset, callbacks=backup_cb, dataset_sink_mode=False)
    # restore
    model.train(1, dataset, callbacks=BackupAndRestore("backup"), dataset_sink_mode=False)


@non_graph_engine
def test_eval():
    """ test_eval """
    dataset_types = (np.float32, np.float32)
    dataset_shapes = ((32, 3, 224, 224), (32, 3))

    dataset = MindData(size=2, batch_size=32,
                       np_types=dataset_types,
                       output_shapes=dataset_shapes,
                       input_indexs=(0, 1))
    net = Net()
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(net, loss_fn=nn.SoftmaxCrossEntropyWithLogits(), metrics={"loss"})
    with pytest.raises(ValueError):
        model.eval(dataset)

    net2 = LossNet()
    model2 = Model(net2, eval_network=net2, eval_indexes=[0, 1, 2], metrics={"loss"})
    with pytest.raises(ValueError):
        model2.eval(dataset)

    _ = LossNet()
    model3 = Model(net2, eval_network=net2, metrics={"loss"})
    with pytest.raises(ValueError):
        model3.eval(dataset)


class TestGraphMode:
    """ TestGraphMode definition """

    def test_train_minddata_graph_mode(self, test_with_simu):
        """ test_train_minddata_graph_mode """
        context.set_context(mode=context.GRAPH_MODE)
        # pylint: disable=unused-argument
        dataset_types = (np.float32, np.float32)
        dataset_shapes = ((32, 3, 224, 224), (32, 3))

        dataset = MindData(size=2, batch_size=32,
                           np_types=dataset_types,
                           output_shapes=dataset_shapes,
                           input_indexs=())
        model = get_model()
        model.train(1, dataset)


class CallbackTest(Callback):
    """ CallbackTest definition """

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *err):
        pass

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print(cb_params.cur_epoch_num, cb_params.cur_step_num)


def test_train_callback(test_with_simu):
    """ test_train_callback """
    context.set_context(mode=context.GRAPH_MODE)
    dataset = get_dataset()
    model = get_model()
    callback = CallbackTest()
    if test_with_simu:
        return
    model.train(2, dataset, callbacks=callback)


log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


# Test the invalid args and trigger RuntimeError
def test_model_build_abnormal_string():
    """ test_model_build_abnormal_string """
    net = nn.ReLU()
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(net)
    err = False
    try:
        model.predict('aaa')
    except TypeError as e:
        log.error("Find type error: %r ", e)
        err = True
    finally:
        assert err


def test_init_model_error():
    """ test_init_model_error """
    net = nn.ReLU()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    with pytest.raises(KeyError):
        Model(net, loss, metrics={"top1"})

    with pytest.raises(ValueError):
        Model(net, metrics={"top_1_accuracy"})

    with pytest.raises(TypeError):
        Model(net, metrics={"top5": None})

    with pytest.raises(ValueError):
        Model(net, eval_network=net, eval_indexes=[], metrics={"top_1_accuracy"})

    with pytest.raises(ValueError):
        Model(net, eval_network=net, eval_indexes=(1, 2, 3), metrics={"top_1_accuracy"})

    with pytest.raises(TypeError):
        Model(net, loss, metrics=("top_1_accuracy"))

    with pytest.raises(TypeError):
        Model(net, loss, metrics=())

    with pytest.raises(TypeError):
        Model(net, loss, metrics=["top_1_accuracy"])


def test_model_eval_error():
    """ test_model_eval_error """
    dataset_types = (np.float32, np.float32)
    dataset_shapes = ((32, 3, 224, 224), (32, 3))

    dataset = MindData(size=2, batch_size=32,
                       np_types=dataset_types,
                       output_shapes=dataset_shapes,
                       input_indexs=())

    net = nn.ReLU()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    context.set_context(mode=context.GRAPH_MODE)
    model_nometrics = Model(net, loss)
    with pytest.raises(ValueError):
        model_nometrics.eval(dataset)

    model_metrics_empty = Model(net, loss, metrics={})
    with pytest.raises(ValueError):
        model_metrics_empty.eval(dataset)
