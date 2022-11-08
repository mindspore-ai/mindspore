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

from mindspore._checkparam import Validator
from mindspore.train.callback._callback import Callback


class TimeMonitor(Callback):
    """
    Monitor the time in train or eval process.

    Args:
        data_size (int): How many steps are the intervals between print information each time.
            if the program get `batch_num` during training, `data_size` will be set to `batch_num`,
            otherwise `data_size` will be used. Default: None.

    Raises:
        ValueError: If data_size is not positive int.

    Examples:
        .. note::
            Before running the following example, you need to customize the network LeNet5 and
            dataset preparation function create_dataset. Refer to
            `Building a Network <https://www.mindspore.cn/tutorials/en/master/beginner/model.html>`_
            and `Dataset <https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html>`_ .

        >>> from mindspore import nn
        >>> from mindspore.train import Model, TimeMonitor
        >>>
        >>> net = LeNet5()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
        >>> data_path = './MNIST_Data'
        >>> dataset = create_dataset(data_path)
        >>> time_monitor = TimeMonitor()
        >>> model.train(10, dataset, callbacks=time_monitor)
    """

    def __init__(self, data_size=None):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_time = time.time()

    def epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        self.epoch_time = time.time()

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
        print("{} epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format
              (mode.title(), epoch_seconds, step_seconds), flush=True)
