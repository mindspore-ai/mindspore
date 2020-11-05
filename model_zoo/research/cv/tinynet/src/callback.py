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
"""custom callbacks for ema and loss"""
from copy import deepcopy

import numpy as np
from mindspore.train.callback import Callback
from mindspore.common.parameter import Parameter
from mindspore.train.serialization import save_checkpoint
from mindspore.nn import Loss, Top1CategoricalAccuracy, Top5CategoricalAccuracy
from mindspore.train.model import Model
from mindspore import Tensor


def load_nparray_into_net(net, array_dict):
    """
    Loads dictionary of numpy arrays into network.

    Args:
        net (Cell): Cell network.
        array_dict (dict): dictionary of numpy array format model weights.
    """
    param_not_load = []
    for _, param in net.parameters_and_names():
        if param.name in array_dict:
            new_param = array_dict[param.name]
            param.set_data(Parameter(Tensor(deepcopy(new_param)), name=param.name))
        else:
            param_not_load.append(param.name)
    return param_not_load


class EmaEvalCallBack(Callback):
    """
    Call back that will evaluate the model and save model checkpoint at
    the end of training epoch.

    Args:
        network: tinynet network instance.
        ema_network: step-wise exponential moving average of network.
        eval_dataset: the evaluation daatset.
        decay (float): ema decay.
        save_epoch (int): defines how often to save checkpoint.
        dataset_sink_mode (bool): whether to use data sink mode.
        start_epoch (int): which epoch to start/resume training.
    """

    def __init__(self, network, ema_network, eval_dataset, loss_fn, decay=0.999,
                 save_epoch=1, dataset_sink_mode=True, start_epoch=0):
        self.network = network
        self.ema_network = ema_network
        self.eval_dataset = eval_dataset
        self.loss_fn = loss_fn
        self.decay = decay
        self.save_epoch = save_epoch
        self.shadow = {}
        self.ema_accuracy = {}

        self.best_ema_accuracy = 0
        self.best_accuracy = 0
        self.best_ema_epoch = 0
        self.best_epoch = 0
        self._start_epoch = start_epoch
        self.eval_metrics = {'Validation-Loss': Loss(),
                             'Top1-Acc': Top1CategoricalAccuracy(),
                             'Top5-Acc': Top5CategoricalAccuracy()}
        self.dataset_sink_mode = dataset_sink_mode

    def begin(self, run_context):
        """Initialize the EMA parameters """
        for _, param in self.network.parameters_and_names():
            self.shadow[param.name] = deepcopy(param.data.asnumpy())

    def step_end(self, run_context):
        """Update the EMA parameters"""
        for _, param in self.network.parameters_and_names():
            new_average = (1.0 - self.decay) * param.data.asnumpy().copy() + \
                self.decay * self.shadow[param.name]
            self.shadow[param.name] = new_average

    def epoch_end(self, run_context):
        """evaluate the model and ema-model at the end of each epoch"""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num + self._start_epoch - 1

        save_ckpt = (cur_epoch % self.save_epoch == 0)
        load_nparray_into_net(self.ema_network, self.shadow)
        model = Model(self.network, loss_fn=self.loss_fn, metrics=self.eval_metrics)
        model_ema = Model(self.ema_network, loss_fn=self.loss_fn,
                          metrics=self.eval_metrics)
        acc = model.eval(
            self.eval_dataset, dataset_sink_mode=self.dataset_sink_mode)
        ema_acc = model_ema.eval(
            self.eval_dataset, dataset_sink_mode=self.dataset_sink_mode)
        print("Model Accuracy:", acc)
        print("EMA-Model Accuracy:", ema_acc)

        output = [{"name": k, "data": Tensor(v)}
                  for k, v in self.shadow.items()]
        self.ema_accuracy[cur_epoch] = ema_acc["Top1-Acc"]
        if self.best_ema_accuracy < ema_acc["Top1-Acc"]:
            self.best_ema_accuracy = ema_acc["Top1-Acc"]
            self.best_ema_epoch = cur_epoch
            save_checkpoint(output, "ema_best.ckpt")

        if self.best_accuracy < acc["Top1-Acc"]:
            self.best_accuracy = acc["Top1-Acc"]
            self.best_epoch = cur_epoch

        print("Best Model Accuracy: %s, at epoch %s" %
              (self.best_accuracy, self.best_epoch))
        print("Best EMA-Model Accuracy: %s, at epoch %s" %
              (self.best_ema_accuracy, self.best_ema_epoch))

        if save_ckpt:
            # Save the ema_model checkpoints
            ckpt = "{}-{}.ckpt".format("ema", cur_epoch)
            save_checkpoint(output, ckpt)
            save_checkpoint(output, "ema_last.ckpt")

            # Save the model checkpoints
            save_checkpoint(cb_params.train_network, "last.ckpt")

        print("Top 10 EMA-Model Accuracies: ")
        count = 0
        for epoch in sorted(self.ema_accuracy, key=self.ema_accuracy.get,
                            reverse=True):
            if count == 10:
                break
            print("epoch: %s, Top-1: %s)" % (epoch, self.ema_accuracy[epoch]))
            count += 1


class LossMonitor(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0, do not print loss.

    Args:
        lr_array (numpy.array): scheduled learning rate.
        total_epochs (int): Total number of epochs for training.
        per_print_times (int): Print the loss every time. Default: 1.
        start_epoch (int): which epoch to start, used when resume from a
        certain epoch.

    Raises:
        ValueError: If print_step is not an integer or less than zero.
    """

    def __init__(self, lr_array, total_epochs, per_print_times=1, start_epoch=0):
        super(LossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self._lr_array = lr_array
        self._total_epochs = total_epochs
        self._start_epoch = start_epoch

    def step_end(self, run_context):
        """log epoch, step, loss and learning rate"""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        cur_epoch_num = cb_params.cur_epoch_num + self._start_epoch - 1
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())
        global_step = cb_params.cur_step_num - 1
        cur_step_in_epoch = global_step % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cur_epoch_num, cur_step_in_epoch))

        if self._per_print_times != 0 and cur_step_in_epoch % self._per_print_times == 0:
            print("epoch: %s/%s, step: %s/%s, loss is %s, learning rate: %s"
                  % (cur_epoch_num, self._total_epochs, cur_step_in_epoch,
                     cb_params.batch_num, loss, self._lr_array[global_step]),
                  flush=True)
