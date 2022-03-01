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
"""LossMonitor Callback class."""

import numpy as np
from mindspore.common.tensor import Tensor

from ._callback import Callback


class LossMonitor(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0, do not print loss.
        Parameter `has_trained_epoch` use for failure recovery scenarios.

    Args:
        per_print_times (int): How many steps to print once loss. During sink mode, it will print loss in the
                               nearest step. Default: 1.
        has_trained_epoch (int): How many epochs has trained. If this parameter is set, LossMonitor will monitor the
                                 loss after has_trained_epoch's epoch. Default: 0.

    Raises:
        ValueError: If per_print_times is not an integer or less than zero.
        ValueError: If has_trained_epoch is not an integer or less than zero.

    Examples:
        >>> from mindspore import Model, nn
        >>>
        >>> net = LeNet5()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
        >>> data_path = './MNIST_Data'
        >>> dataset = create_dataset(data_path)
        >>> loss_monitor = LossMonitor()
        >>> model.train(10, dataset, callbacks=loss_monitor)
    """

    def __init__(self, per_print_times=1, has_trained_epoch=0):
        super(LossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("The argument 'per_print_times' must be int and >= 0, "
                             "but got {}".format(per_print_times))
        if not isinstance(has_trained_epoch, int) or has_trained_epoch < 0:
            raise ValueError("The argument 'has_trained_epoch' must be int and >= 0, "
                             "but got {}".format(has_trained_epoch))
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self._has_trained_epoch = has_trained_epoch

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch + self._has_trained_epoch))
        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num + self._has_trained_epoch,
                                                      cur_step_in_epoch, loss), flush=True)
