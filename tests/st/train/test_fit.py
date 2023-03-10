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
import sys
import re

import pytest
import numpy as np

import mindspore as ms
from mindspore import Model, nn
from mindspore.train.callback import LossMonitor
from mindspore import dataset as ds


def get_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = 0
        y = x * w + b
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)


def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size, drop_remainder=True)
    input_data = input_data.repeat(repeat_size)
    return input_data


def define_model():
    net = nn.Dense(1, 1, has_bias=False)
    net_loss = nn.MSELoss()
    net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
    return Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={'mse', 'mae'})


class Redirect:
    """
    Get the content of callbacks.
    """
    content = ""

    def write(self, str1):
        self.content = str1 + self.content

    def flush(self):
        self.content = ""


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fit_train_dataset_non_sink_mode(mode):
    """
    Feature: `mindspore.train.Model.fit` with train dataset in non-sink mode.
    Description: test fit with train dataset in non-sink mode.
    Expectation: run in non-sink mode.
    """
    ms.set_context(mode=mode)
    model = define_model()
    ds_train = create_dataset(4096, 1024)
    ds_eval = create_dataset(1024, 512)
    callbacks = [LossMonitor()]
    r = Redirect()
    current = sys.stdout
    sys.stdout = r
    model.fit(1, ds_train, ds_eval, callbacks=callbacks, dataset_sink_mode=False)
    sys.stdout = current
    assert re.search("'mse': 9.0", r.content)
    assert re.search("'mae': 3.0", r.content)
    r.flush()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fit_train_dataset_sink_mode(mode):
    """
    Feature: `mindspore.train.Model.fit` with train dataset in sink mode.
    Description: test fit with train dataset in sink mode.
    Expectation: run in sink mode.
    """
    ms.set_context(mode=mode)
    model = define_model()
    ds_train = create_dataset(4096, 1024)
    ds_eval = create_dataset(1024, 512)
    callbacks = [LossMonitor()]
    r = Redirect()
    current = sys.stdout
    sys.stdout = r
    model.fit(1, ds_train, ds_eval, callbacks=callbacks, dataset_sink_mode=True, sink_size=256)
    sys.stdout = current
    assert re.search("'mse': 9.0", r.content)
    assert re.search("'mae': 3.0", r.content)
    r.flush()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fit_valid_dataset_non_sink_mode(mode):
    """
    Feature: `mindspore.train.Model.fit` with valid dataset in non-sink mode.
    Description: test fit with valid dataset in non-sink mode.
    Expectation: run in non-sink mode.
    """
    ms.set_context(mode=mode)
    model = define_model()
    ds_train = create_dataset(4096, 1024)
    ds_eval = create_dataset(1024, 512)
    callbacks = [LossMonitor()]
    r = Redirect()
    current = sys.stdout
    sys.stdout = r
    model.fit(1, ds_train, ds_eval, callbacks=callbacks, valid_dataset_sink_mode=False)
    sys.stdout = current
    assert re.search("'mse': 9.0", r.content)
    assert re.search("'mae': 3.0", r.content)
    r.flush()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fit_valid_dataset_sink_mode(mode):
    """
    Feature: `mindspore.train.Model.fit` with valid dataset in sink mode.
    Description: test fit with valid dataset in sink mode.
    Expectation: run in sink mode.
    """
    ms.set_context(mode=mode)
    model = define_model()
    ds_train = create_dataset(4096, 1024)
    ds_eval = create_dataset(1024, 512)
    callbacks = [LossMonitor()]
    r = Redirect()
    current = sys.stdout
    sys.stdout = r
    model.fit(1, ds_train, ds_eval, callbacks=callbacks, valid_dataset_sink_mode=True)
    sys.stdout = current
    assert re.search("'mse': 9.0", r.content)
    assert re.search("'mae': 3.0", r.content)
    r.flush()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fit_valid_frequency(mode):
    """
    Feature: check `valid_frequency` input  in `mindspore.train.Model.fit`.
    Description: when `valid_frequency` is integer, list or other types.
    Expectation: Executed fit valid frequency successfully.
    """
    ms.set_context(mode=mode)
    model = define_model()
    callbacks = [LossMonitor()]
    ds_train = create_dataset(4096, 1024)
    ds_eval = create_dataset(1024, 512)
    r = Redirect()
    current = sys.stdout
    sys.stdout = r
    model.fit(4, ds_train, ds_eval, valid_frequency=2, callbacks=callbacks)
    sys.stdout = current
    assert re.search("Eval result: epoch 4", r.content)
    assert re.search("'mse': 9.0", r.content)
    assert re.search("'mae': 3.0", r.content)
    r.flush()
