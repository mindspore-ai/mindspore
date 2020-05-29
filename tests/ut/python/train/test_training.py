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
from mindspore.nn.optim import Momentum
from mindspore.train.callback import SummaryStep
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
        # pylint: disable=unused-argument
        dataset_types = (np.float32, np.float32)
        dataset_shapes = ((32, 3, 224, 224), (32, 3))

        dataset = MindData(size=2, batch_size=32,
                           np_types=dataset_types,
                           output_shapes=dataset_shapes,
                           input_indexs=())
        model = get_model()
        model.train(1, dataset)


class CallbackTest:
    """ CallbackTest definition """

    def __init__(self):
        pass

    def record(self, step, *args):
        print(step, args)


def test_train_callback(test_with_simu):
    """ test_train_callback """
    dataset = get_dataset()
    model = get_model()
    fn = CallbackTest()
    summary_recode = SummaryStep(fn, 2)
    if test_with_simu:
        return
    model.train(2, dataset, callbacks=summary_recode)


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
    except ValueError as e:
        log.error("Find value error: %r ", e)
        err = True
    finally:
        assert err


def test_model_init():
    """ test_model_init_error """
    train_dataset = get_dataset()
    eval_dataset = get_dataset()

    with pytest.raises(RuntimeError):
        context.set_context(mode=context.PYNATIVE_MODE)
        get_model().init(train_dataset)

    context.set_context(mode=context.GRAPH_MODE)
    get_model().init(train_dataset)
    get_model(metrics={'acc'}).init(eval_dataset)

    with pytest.raises(RuntimeError):
        get_model().init(train_dataset, eval_dataset)
    with pytest.raises(ValueError):
        get_model().init()


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
