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
"""TimeMonitor Callback class."""
from __future__ import absolute_import

import time

from mindspore import _checkparam as Validator
from mindspore.train.callback._callback import Callback


class TimeMonitor(Callback):
    """
    Monitor the time in train or eval process.

    Args:
        data_size (int): How many steps are the intervals between print information each time.
            if the program get `batch_num` during training, `data_size` will be set to `batch_num`,
            otherwise `data_size` will be used. Default: ``None`` .

        data_time (bool): Whether to sow the average time of fetching data in Host.
            Note that data fetch and network compute are processed sequentially in non dataset sink mode, while
            they are asynchronous in dataset sink mode. Default: ``False`` .

    Raises:
        ValueError: If data_size is not positive int.
        TypeError: If data_time is not bool.

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.train import Model, TimeMonitor
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
        >>> # Create the dataset taking MNIST as an example. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py
        >>> dataset = create_dataset()
        >>> time_monitor = TimeMonitor()
        >>> model.train(10, dataset, callbacks=time_monitor)
    """

    def __init__(self, data_size=None, data_time=False):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_time = time.time()
        self.data_time = data_time
        self.data_time_sum = 0.0
        self.data_time_start = 0.0
        self.data_sink = lambda c: c.original_args()["dataset_sink_mode"]
        Validator.check_bool(data_time, "data_time")

    def on_train_step_begin(self, run_context):
        """
        Record time at the beginning of step.

        Args:
            run_context (RunContext): Context of the process running. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        if self.data_time and not self.data_sink(run_context):
            interval = time.time() - self.data_time_start
            self.data_time_sum = self.data_time_sum + interval

    def on_train_step_end(self, run_context):
        """
        Record time at the end of step.

        Args:
            run_context (RunContext): Context of the process running. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        if self.data_time and not self.data_sink(run_context):
            self.data_time_start = time.time()

    def epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        self.epoch_time = time.time()
        if self.data_time and not self.data_sink(run_context):
            self.data_time_sum = 0.0
            self.data_time_start = time.time()

    def epoch_end(self, run_context):
        """
        Print process cost time at the end of epoch.

        Args:
           run_context (RunContext): Context of the process running. For more details,
                   please refer to :class:`mindspore.train.RunContext`.
        """
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        mode = cb_params.get("mode", "")
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num
        Validator.check_positive_int(step_size)

        step_seconds = epoch_seconds / step_size

        train_log = "{} epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format(
            mode.title(), epoch_seconds, step_seconds)

        if self.data_time and not self.data_sink(run_context):
            data_step_seconds = self.data_time_sum * 1000 / step_size
            data_log = " (data time: {:5.3f} ms)".format(data_step_seconds)
            train_log += data_log
        elif self.data_time and self.data_sink(run_context):
            # send info viewer to query epoch message of cur_epoch_num
            send_info = cb_params["dataset_helper"].get_send_info(run_context)
            epoch = cb_params["cur_epoch_num"]
            epoch_send_info = send_info.epoch(epoch)
            # show average time of fetching data time
            fetch_data_time = epoch_send_info["fetch_data_time"]
            data_log = " (data time: {:5.3f} ms)".format(fetch_data_time)
            train_log += data_log

        print(train_log, flush=True)
