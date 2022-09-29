# Copyright 2022 Huawei Technologies Co., Ltd
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

""" test_fit """

import pytest
import numpy as np
from mindspore import Model, nn, Tensor
from mindspore.common.initializer import Normal
from mindspore.train.callback import Callback, TimeMonitor, LossMonitor
from mindspore import dataset as ds


def get_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)


def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size, drop_remainder=True)
    input_data = input_data.repeat(repeat_size)
    return input_data


def define_model():
    net = nn.Dense(1, 1, Normal(0.02), Normal(0.02))
    net_loss = nn.MSELoss()
    net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
    return Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={'mse', 'mae'})


class MyCallbackOldMethod(Callback):
    """
    Raise warning in `mindspore.train.Model.train` and  `mindspore.train.Model.eval`;
    raise error  in `mindspore.train.Model.fit`.
    """
    def begin(self, run_context):
        print("custom callback: print on begin, just for test.")

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        outputs = cb_params.get("net_outputs")
        result = outputs if isinstance(outputs, Tensor) else outputs[0]
        print("custom train callback: step end, loss is %s" % (result))

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        print("custom train callback: epoch end, loss is %s" % (cb_params.get("net_outputs")))


class MyCallbackNewMethod(Callback):
    """
    Custom callback running in `mindspore.train.Model.train`, `mindspore.train.Model.eval`,
    `mindspore.train.Model.fit`.
    """
    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        print("custom callback: train epoch end, loss is %s" % (cb_params.get("net_outputs")))

    def on_eval_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        print("custom callback: eval epoch end, metric is %s" % (cb_params.get("net_outputs")[0]))


def test_fit_train_dataset_non_sink_mode():
    """
    Feature: `mindspore.train.Model.fit` with train dataset in non-sink mode.
    Description: test fit with train dataset in non-sink mode.
    Expectation: run in non-sink mode.
    """
    model = define_model()
    ds_train = create_dataset(4096, 1024)
    ds_eval = create_dataset(1024, 512)
    callbacks = [LossMonitor()]
    model.fit(3, ds_train, ds_eval, callbacks=callbacks, dataset_sink_mode=False)


def test_fit_train_dataset_sink_mode():
    """
    Feature: `mindspore.train.Model.fit` with train dataset in sink mode.
    Description: test fit with train dataset in sink mode.
    Expectation: run in sink mode.
    """
    model = define_model()
    ds_train = create_dataset(4096, 1024)
    ds_eval = create_dataset(1024, 512)
    callbacks = [LossMonitor()]
    model.fit(3, ds_train, ds_eval, callbacks=callbacks, dataset_sink_mode=True, sink_size=256)


def test_fit_valid_dataset_non_sink_mode():
    """
    Feature: `mindspore.train.Model.fit` with valid dataset in non-sink mode.
    Description: test fit with valid dataset in non-sink mode.
    Expectation: run in non-sink mode.
    """
    model = define_model()
    ds_train = create_dataset(4096, 1024)
    ds_eval = create_dataset(1024, 512)
    callbacks = [LossMonitor()]
    model.fit(3, ds_train, ds_eval, callbacks=callbacks, valid_dataset_sink_mode=False)


def test_fit_valid_dataset_sink_mode():
    """
    Feature: `mindspore.train.Model.fit` with valid dataset in sink mode.
    Description: test fit with valid dataset in sink mode.
    Expectation: run in sink mode.
    """
    model = define_model()
    ds_train = create_dataset(4096, 1024)
    ds_eval = create_dataset(1024, 512)
    callbacks = [LossMonitor()]
    model.fit(3, ds_train, ds_eval, callbacks=callbacks, valid_dataset_sink_mode=True)


def test_fit_without_valid_dataset():
    """
    Feature: `mindspore.train.Model.fit` without `valid_dataset` input .
    Description: test fit when `valid_dataset` is None and `valid_dataset_sink_mode` is True or False.
    Expectation: network train without eval process, `valid_dataset_sink_mode` does not take effect.
    """
    model = define_model()
    ds_train = create_dataset(4096, 1024)
    callbacks = [LossMonitor()]
    model.fit(3, ds_train, None, callbacks=callbacks, valid_dataset_sink_mode=False)
    model.fit(3, ds_train, None, callbacks=callbacks)


def test_fit_valid_frequency():
    """
    Feature: check `valid_frequency` input  in `mindspore.train.Model.fit`.
    Description: when `valid_frequency` is integer, list or other types.
    Expectation: raise ValueError when the type of valid_frequency is not int or list.
    """
    model = define_model()
    callbacks = [LossMonitor()]
    ds_train = create_dataset(4096, 1024)
    ds_eval = create_dataset(1024, 512)
    model.fit(3, ds_train, ds_eval, valid_frequency=1, callbacks=callbacks)
    model.fit(5, ds_train, ds_eval, valid_frequency=2, callbacks=callbacks)
    model.fit(5, ds_train, ds_eval, valid_frequency=[0, 1, 4], callbacks=callbacks)
    with pytest.raises(ValueError):
        model.fit(5, ds_train, ds_eval, valid_frequency=(0, 2), callbacks=callbacks)


def test_fit_callbacks():
    """
    Feature: check `callbacks` input in `mindspore.train.Model.fit`.
    Description: test internal or custom callbacks in fit.
    Expectation: raise ValueError when methods of custom callbacks are not prefixed with 'on_train' or  'on_eval'.
    """
    model = define_model()
    ds_train = create_dataset(4096, 1024)
    ds_eval = create_dataset(1024, 512)
    model.fit(3, ds_train, ds_eval, callbacks=None)
    model.fit(3, ds_train, ds_eval, callbacks=[TimeMonitor()])
    model.fit(3, ds_train, ds_eval, callbacks=[TimeMonitor(), LossMonitor()])
    model.fit(3, ds_train, ds_eval, callbacks=[MyCallbackNewMethod()])
    model.fit(3, ds_train, ds_eval, callbacks=[TimeMonitor(), MyCallbackNewMethod()])
    with pytest.raises(ValueError):
        model.fit(3, ds_train, ds_eval, callbacks=[MyCallbackOldMethod()])
    with pytest.raises(ValueError):
        model.fit(3, ds_train, ds_eval, callbacks=[TimeMonitor(), MyCallbackOldMethod()])
    with pytest.raises(ValueError):
        model.fit(3, ds_train, valid_dataset=None, callbacks=[TimeMonitor(), MyCallbackOldMethod()])


def test_train_eval_callbacks():
    """
    Feature: check `callbacks` input in `mindspore.train.Model.train` or `mindspore.train.Model.eval`.
    Description: test internal or custom callbacks in train or eval.
    Expectation: raise warning when methods of custom callbacks are not prefixed with 'on_train' or  'on_eval'.
    """
    model = define_model()
    ds_train = create_dataset(4096, 1024)
    ds_eval = create_dataset(1024, 512)

    model.train(3, ds_train, callbacks=None)
    model.train(3, ds_train, callbacks=[TimeMonitor()])
    model.train(3, ds_train, callbacks=[LossMonitor()])
    model.train(3, ds_train, callbacks=[MyCallbackNewMethod()])
    model.train(3, ds_train, callbacks=[MyCallbackOldMethod()])

    metric_results = model.eval(ds_eval, callbacks=None)
    print("{}".format(metric_results))
    metric_results = model.eval(ds_eval, callbacks=[TimeMonitor()])
    print("{}".format(metric_results))
    metric_results = model.eval(ds_eval, callbacks=[MyCallbackNewMethod()])
    print("{}".format(metric_results))
    metric_results = model.eval(ds_eval, callbacks=[MyCallbackOldMethod()])
    print("{}".format(metric_results))
