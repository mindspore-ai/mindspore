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
"""Test callback."""
import numpy as np
import mindspore.ops.operations as P
from mindspore import nn, Tensor
from mindspore.train import Model
from mindspore.nn import Momentum, Conv2d
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import History
from mindspore.train.callback import LambdaCallback
from mindspore.train.callback import LossMonitor
from tests.st.frontend.dataset.animal import create_animal_no_random_dataset


class TrainMeNet(nn.Cell):
    def __init__(self, weight_init):
        super().__init__()
        self.conv = Conv2d(in_channels=3, out_channels=12, kernel_size=1, weight_init=weight_init)
        self.reduce = P.ReduceMean()

    def construct(self, inputs):
        out = self.conv(inputs)
        out = self.reduce(out, (2, 3))
        return out


class CallbackFactory:
    def __init__(self):
        super().__init__()
        self.weight_init = np.random.randn(12, 3, 1, 1).astype(np.float32) * 0.01

    def callback_history(self, metrics_param=None, iftrain=True, epochs=3):
        net = TrainMeNet(weight_init=Tensor(self.weight_init))
        ds_train = create_animal_no_random_dataset(epoch_size=epochs)
        ds_eval = create_animal_no_random_dataset(epoch_size=epochs)

        loss = SoftmaxCrossEntropyWithLogits(sparse=False)
        opt = Momentum(learning_rate=0.1, momentum=0.9, params=net.get_parameters())
        cb1 = History()
        cb2 = LossMonitor()
        model = Model(net, loss, opt, metrics=metrics_param)
        if iftrain is True:
            model.train(epochs, ds_train, callbacks=[cb2, cb1], dataset_sink_mode=True)
            assert len(cb1.history['net_output']) == epochs
            assert len(cb1.epoch['epoch']) == epochs
            assert cb1.epoch['epoch'] == [1, 2, 3]
        else:
            model.eval(ds_eval, callbacks=[cb1], dataset_sink_mode=True)
            assert len(cb1.history['net_output']) == 1
            assert len(cb1.epoch['epoch']) == 1

    def callback_lambdacallback(self, metrics_param=None, iftrain=True, epochs=3):
        net = TrainMeNet(weight_init=Tensor(self.weight_init))
        ds_train = create_animal_no_random_dataset(epoch_size=epochs)
        ds_eval = create_animal_no_random_dataset(epoch_size=epochs)

        loss = SoftmaxCrossEntropyWithLogits(sparse=False)
        opt = Momentum(learning_rate=0.1, momentum=0.9, params=net.get_parameters())
        cb1 = LambdaCallback(
            on_train_step_end=lambda run_context: print('loss: ',
                                                        run_context.original_args().net_outputs),
            on_eval_step_end = lambda run_context: print('loss: ',
                                                        run_context.original_args().net_outputs))
        cb2 = LambdaCallback(
            on_train_epoch_end =lambda run_context: print('epoch: ',
                                                run_context.original_args().epoch_num + 1),
            on_eval_epoch_end=lambda run_context: print('epoch: ',
                                                run_context.original_args().epoch_num + 1))
        model = Model(net, loss, opt, metrics=metrics_param)
        if iftrain is True:
            model.train(epochs, ds_train, callbacks=[cb1, cb2], dataset_sink_mode=True)
        else:
            model.eval(ds_eval, callbacks=[cb1], dataset_sink_mode=True)


def test_lambdacallback_train_metrics():
    """
    Feature: test callback.
    Description: test callback.
    Expectation: the result match with expected result.
    """
    fact = CallbackFactory()
    metrics_param_me = {"loss", "top_1_accuracy", "top_5_accuracy"}
    fact.callback_lambdacallback(metrics_param_me)


def test_lambdacallback_eval_metrics():
    """
    Feature: test callback.
    Description: test callback.
    Expectation: the result match with expected result.
    """
    fact = CallbackFactory()
    metrics_param_me = {"loss", "top_1_accuracy", "top_5_accuracy"}
    fact.callback_lambdacallback(metrics_param_me, iftrain=False)
