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

""" test EarlyStopping and ReduceLROnPlateau Callbacks"""

import copy
import pytest
import numpy as np

import mindspore as ms
from mindspore import nn, Model
from mindspore import dataset as ds
from mindspore.nn.optim import Momentum
from mindspore.common.initializer import Normal
from mindspore.train import RunContext, ReduceLROnPlateau, EarlyStopping
from mindspore.train.callback import _InternalCallbackParam, _CallbackManager


def get_data(num, w=4.0, b=5.0):
    for _ in range(num):
        x = np.random.uniform(-5.0, 5.0)
        value = (x * x - x * w + b + np.random.normal(0, 1)) // 12
        target_onehot = np.zeros(shape=(5,))
        target_onehot[int(value)] = 1
        yield np.array([x]).astype(np.float32), target_onehot.astype(np.float32)


def create_dataset(num_data, batch_size=512, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data


def define_model(metrics):
    net = nn.Dense(1, 5, Normal(0.02))
    net_loss = nn.SoftmaxCrossEntropyWithLogits()
    net_opt = nn.Momentum(net.trainable_params(), 0.1, 0.9)
    model = Model(net, loss_fn=net_loss, optimizer=net_opt, metrics=metrics)
    return model


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_reduce_lr_on_plateau_patience_and_cooldown(mode):
    """
    Feature: `patience` and `cooldown`.
    Description: test whether the learning rate reduces correct.
    Expectation: output learning rates match the expectation lrs.
    """
    ms.set_context(mode=mode)
    net = nn.Dense(1, 1, Normal(0.02), Normal(0.02))
    cb_params = _InternalCallbackParam()
    run_context = RunContext(cb_params)

    cases = [
        {"losses": [1.0, 1.1, 1.2], "patience": 2, "cooldown": 0, "lrs": [1.0, 1.0, 0.1]},
        {"losses": [1.0, 1.1, 0.9, 1.0, 1.1], "patience": 2, "cooldown": 0, "lrs": [1.0, 1.0, 1.0, 1.0, 0.1]},
        {"losses": [1.0, 1.1, 1.0, 1.0, 1.1], "patience": 2, "cooldown": 0, "lrs": [1.0, 1.0, 0.1, 0.1, 0.01]},
        {"losses": [1.0, 1.1, 1.0, 1.0, 1.1, 1.2], "patience": 2, "cooldown": 1,
         "lrs": [1.0, 1.0, 0.1, 0.1, 0.01, 0.01]},
        {"losses": [1.0, 1.1, 1.0, 1.0, 1.1, 1.2], "patience": 2, "cooldown": 2,
         "lrs": [1.0, 1.0, 0.1, 0.1, 0.1, 0.01]}
    ]

    for case_i, current_case in enumerate(cases):
        cb_params.optimizer = Momentum(net.trainable_params(), learning_rate=1.0, momentum=0.9)

        losses, patience, cooldown, lrs_results = current_case["losses"], current_case["patience"], \
                                          current_case["cooldown"], current_case["lrs"]

        eval_results = [{'eval_loss': losses[i]} for i in range(len(losses))]
        callbacks = [ReduceLROnPlateau(monitor='eval_loss', patience=patience, cooldown=cooldown)]
        lrs = []
        with _CallbackManager(callbacks) as callbacklist:
            for i, result in enumerate(eval_results):
                callbacklist.on_train_epoch_begin(run_context)
                cb_params.eval_results = result
                cb_params.cur_epoch_num = i + 1
                callbacklist.on_train_epoch_end(run_context)
                cur_lr = cb_params.optimizer.learning_rate.asnumpy()
                lrs.append(copy.deepcopy(cur_lr))
        assert np.allclose(lrs, lrs_results[case_i], atol=1e-7)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_earlystopping_with_baseline(mode):
    """
    Feature: `baseline` in EarlyStopping.
    Description: test whether the stopped epoch correct.
    Expectation: the stopped epoch match the expectation stop_epoch.
    """
    ms.set_context(mode=mode)
    cases = [
        {"baseline": 0.3, "accuracy": [0.6, 0.5, 0.7, 0.5, 0.6], "patience": 2, "stop_epoch": 5},
        {"baseline": 0.55, "accuracy": [0.6, 0.3, 0.5, 0.5], "patience": 2, "stop_epoch": 3},
        {"baseline": 0.6, "accuracy": [0.5, 0.4, 0.7, 0.6, 0.5, 0.6], "patience": 3, "stop_epoch": 6},
    ]
    for _, current_case in enumerate(cases):
        baseline, acc, patience, stop_epoch = current_case["baseline"], current_case["accuracy"], \
                                          current_case["patience"], current_case["stop_epoch"]

        eval_results = [{'accuracy': acc[i]} for i in range(len(acc))]
        callbacks = [EarlyStopping(monitor='accuracy', patience=patience, baseline=baseline, verbose=True)]

        cb_params = _InternalCallbackParam()
        run_context = RunContext(cb_params)

        with _CallbackManager(callbacks) as callbacklist:
            for i, result in enumerate(eval_results):
                callbacklist.on_train_epoch_begin(run_context)
                cb_params.eval_results = result
                cb_params.cur_epoch_num = i+1
                callbacklist.on_train_epoch_end(run_context)
                if run_context.get_stop_requested():
                    break
            callbacklist.on_train_end(run_context)
            cur_epoch = cb_params.cur_epoch_num
            assert cur_epoch == stop_epoch


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_earlystopping_final_weights_when_restoring_model_weights(mode):
    """
    Feature: `restore_best_weights` in EarlyStopping.
    Description: test whether the model weights saved is correct.
    Expectation: Giving monitor varies as `losses`, the training process is
    expected to be stopped at 3rd epoch, restores the weights of the 2nd epoch.
    """
    ms.set_context(mode=mode)
    callbacks = EarlyStopping(patience=1, monitor="eval_loss", verbose=True, restore_best_weights=True)
    ds_train = create_dataset(1024, 512)
    model_train = define_model(metrics={"acc"})

    losses = [1.0, 0.8, 1.2, 1.3, 1.4]
    eval_results = [{'eval_loss': losses[i]} for i in range(len(losses))]

    cb_params = _InternalCallbackParam()
    cb_params.train_network = model_train.train_network
    with _CallbackManager(callbacks) as list_callback:
        run_context = RunContext(cb_params)
        list_callback.on_train_begin(run_context)
        for i in range(5):
            list_callback.on_train_epoch_begin(run_context)
            cb_params.cur_epoch_num = i + 1
            for d in ds_train.create_dict_iterator():
                cb_params.train_network(d["data"], d["label"])
            if cb_params.cur_epoch_num == 2:
                best_net_param_dict = copy.deepcopy(cb_params.train_network.parameters_dict())
            cb_params.eval_results = eval_results[i]
            list_callback.on_train_epoch_end(run_context)
            end_net_param_dict = copy.deepcopy(cb_params.train_network.parameters_dict())
            should_stop = run_context.get_stop_requested()
            if should_stop:
                break
        list_callback.on_train_end(run_context)

    for key in ["weight", "bias"]:
        assert (best_net_param_dict[key].asnumpy() == end_net_param_dict[key].asnumpy()).all()
