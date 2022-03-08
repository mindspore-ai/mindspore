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

import numpy as np
from mindspore.common.tensor import Tensor
from ._callback import Callback

class History(Callback):
    """
    Records the network outputs information into a `History` object.

    The network outputs information will be the loss value if not custimizing the train network or eval network;
    if the custimized network returns a `Tensor` or `numpy.ndarray`, the mean value of network output
    will be recorded, if the custimized network returns a `tuple` or `list`, the first element of network
    outputs will be recorded.

    Note:
        Normally used in `mindspore.Model.train`.

    Examples:
        >>> from mindspore import Model, nn
        >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
        >>> train_dataset = ds.NumpySlicesDataset(data=data).batch(32)
        >>> net = nn.Dense(10, 5)
        >>> crit = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> history_cb = History()
        >>> model = Model(network=net, optimizer=opt, loss_fn=crit, metrics={"recall"})
        >>> model.train(2, train_dataset, callbacks=[history_cb])
        >>> print(history_cb.epoch)
        >>> print(history_cb.history)
        [1, 2]
        {'net_output': [1.607877, 1.6033841]}
    """
    def __init__(self):
        super(History, self).__init__()
        self.history = {}

    def begin(self, run_context):
        """
        Initialize the `epoch` property at the begin of training.

        Args:
            run_context (RunContext): Context of the `mindspore.Model.train/eval`.
        """
        self.epoch = []

    def epoch_end(self, run_context):
        """
        Records the first element of network outputs at the end of epoch.

        Args:
            run_context (RunContext): Context of the `mindspore.Model.train/eval`.
        """
        cb_params = run_context.original_args()
        epoch = cb_params.get("cur_epoch_num", 1)
        self.epoch.append(epoch)
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
