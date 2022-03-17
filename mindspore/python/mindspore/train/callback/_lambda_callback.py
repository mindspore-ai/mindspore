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
"""Lambda Callback class."""

from ._callback import Callback

class LambdaCallback(Callback):
    """
    Callback for creating simple, custom callbacks.

    This callback is constructed with anonymous functions that will be called
    at the appropriate time (during `mindspore.Model.{train | eval}`).

    Note that each stage of callbacks expects one positional arguments: `run_context`.

    Args:
        epoch_begin (Function): called at the beginning of every epoch.
        epoch_end (Function): called at the end of every epoch.
        step_begin (Function): called at the beginning of every batch.
        step_end (Function): called at the end of every batch.
        begin (Function): called at the beginning of model train/eval.
        end (Function): called at the end of model train/eval.

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> from mindspore.train.callback import LambdaCallback
        >>> from mindspore import Model, nn
        >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
        >>> train_dataset = ds.NumpySlicesDataset(data=data).batch(32)
        >>> net = nn.Dense(10, 5)
        >>> crit = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> lambda_callback = LambdaCallback(epoch_end=
        ... lambda run_context: print("loss: ", run_context.original_args().net_outputs))
        >>> model = Model(network=net, optimizer=opt, loss_fn=crit, metrics={"recall"})
        >>> model.train(2, train_dataset, callbacks=[lambda_callback])
        loss: 1.6127687
        loss: 1.6106578
    """
    def __init__(self, epoch_begin=None, epoch_end=None, step_begin=None,
                 step_end=None, begin=None, end=None):
        super(LambdaCallback, self).__init__()
        self.epoch_begin = epoch_begin if epoch_begin else lambda run_context: None
        self.epoch_end = epoch_end if epoch_end else lambda run_context: None
        self.step_begin = step_begin if step_begin else lambda run_context: None
        self.step_end = step_end if step_end else lambda run_context: None
        self.begin = begin if begin else lambda run_context: None
        self.end = end if end else lambda run_context: None
