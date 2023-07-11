# Copyright 2021 Huawei Technologies Co., Ltd
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
"""History Callback class."""
from __future__ import absolute_import

import numpy as np

from mindspore.common.tensor import Tensor
from mindspore.train.callback._callback import Callback


class History(Callback):
    """
    Records the network outputs and metrics information into a `History` object.

    The network outputs information will be the loss value if not custimizing the train network or eval network;
    if the custimized network returns a `Tensor` or `numpy.ndarray`, the mean value of network output
    will be recorded, if the custimized network returns a `tuple` or `list`, the first element of network
    outputs will be recorded.

    Note:
        Normally used in `mindspore.train.Model.train` or `mindspore.train.Model.fit`.

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> from mindspore import nn
        >>> from mindspore.train import Model, History
        >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
        >>> train_dataset = ds.NumpySlicesDataset(data=data).batch(32)
        >>> net = nn.Dense(10, 5)
        >>> crit = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> history_cb = History()
        >>> model = Model(network=net, optimizer=opt, loss_fn=crit, metrics={"recall"})
        >>> model.train(2, train_dataset, callbacks=[history_cb])
        >>> print(history_cb.epoch)
        {'epoch': [1, 2]}
        >>> print(history_cb.history)
        {'net_output': [1.607877, 1.6033841]}
    """
    def __init__(self):
        super(History, self).__init__()
        self.history = {}
        self.epoch = None

    def begin(self, run_context):
        """
        Initialize the `epoch` property at the begin of training.

        Args:
            run_context (RunContext): Context of the `mindspore.train.Model.{train | eval}`. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        self.epoch = {"epoch": []}

    def epoch_end(self, run_context):
        """
        Records the first element of network outputs and metrics information at the end of epoch.

        Args:
            run_context (RunContext): Context of the `mindspore.train.Model.{train | eval}`.  For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        cb_params = run_context.original_args()
        epoch = cb_params.get("cur_epoch_num", 1)
        self.epoch.get("epoch").append(epoch)
        net_output = cb_params.net_outputs
        if isinstance(net_output, (tuple, list)):
            if isinstance(net_output[0], Tensor) and isinstance(net_output[0].asnumpy(), np.ndarray):
                net_output = net_output[0]
        if isinstance(net_output, Tensor) and isinstance(net_output.asnumpy(), np.ndarray):
            net_output = np.mean(net_output.asnumpy())

        metrics = cb_params.get("metrics")
        cur_history = {"net_output": net_output}
        if metrics:
            cur_history.update(metrics)
        for k, v in cur_history.items():
            self.history.setdefault(k, []).append(v)
