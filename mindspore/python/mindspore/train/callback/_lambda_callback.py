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
from __future__ import absolute_import

from mindspore.train.callback._callback import Callback


class LambdaCallback(Callback):
    """
    Callback for creating simple, custom callbacks.

    This callback is constructed with anonymous functions that will be called
    at the appropriate time (during `mindspore.train.Model.{train | eval | fit}`). Note that
    each stage of callbacks expects one positional arguments: `run_context`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        on_train_epoch_begin (Function): called at each train epoch begin. Default: ``None`` .
        on_train_epoch_end (Function): called at each train epoch end. Default: ``None`` .
        on_train_step_begin (Function):  called at each train step begin. Default: ``None`` .
        on_train_step_end (Function): called at each train step end. Default: ``None`` .
        on_train_begin (Function): called at the beginning of model train. Default: ``None`` .
        on_train_end (Function): called at the end of model train. Default: ``None`` .
        on_eval_epoch_begin (Function): called at eval epoch begin. Default: ``None`` .
        on_eval_epoch_end (Function): called at eval epoch end. Default: ``None`` .
        on_eval_step_begin (Function): called at each eval step begin. Default: ``None`` .
        on_eval_step_end (Function): called at each eval step end. Default: ``None`` .
        on_eval_begin (Function): called at the beginning of model eval. Default: ``None`` .
        on_eval_end (Function): called at the end of model eval. Default: ``None`` .

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> from mindspore import nn
        >>> from mindspore.train import Model, LambdaCallback
        >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
        >>> train_dataset = ds.NumpySlicesDataset(data=data).batch(32)
        >>> net = nn.Dense(10, 5)
        >>> crit = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> lambda_callback = LambdaCallback(on_train_epoch_end=
        ... lambda run_context: print("loss:", run_context.original_args().net_outputs))
        >>> model = Model(network=net, optimizer=opt, loss_fn=crit, metrics={"recall"})
        >>> model.train(2, train_dataset, callbacks=[lambda_callback])
        loss: 1.6127687
        loss: 1.6106578
    """
    def __init__(self, on_train_epoch_begin=None, on_train_epoch_end=None, on_train_step_begin=None,
                 on_train_step_end=None, on_train_begin=None, on_train_end=None,
                 on_eval_epoch_begin=None, on_eval_epoch_end=None, on_eval_step_begin=None,
                 on_eval_step_end=None, on_eval_begin=None, on_eval_end=None):
        super(LambdaCallback, self).__init__()
        self.on_train_epoch_begin = on_train_epoch_begin if on_train_epoch_begin else lambda run_context: None
        self.on_train_epoch_end = on_train_epoch_end if on_train_epoch_end else lambda run_context: None
        self.on_train_step_begin = on_train_step_begin if on_train_step_begin else lambda run_context: None
        self.on_train_step_end = on_train_step_end if on_train_step_end else lambda run_context: None
        self.on_train_begin = on_train_begin if on_train_begin else lambda run_context: None
        self.on_train_end = on_train_end if on_train_end else lambda run_context: None

        self.on_eval_epoch_begin = on_eval_epoch_begin if on_eval_epoch_begin else lambda run_context: None
        self.on_eval_epoch_end = on_eval_epoch_end if on_eval_epoch_end else lambda run_context: None
        self.on_eval_step_begin = on_eval_step_begin if on_eval_step_begin else lambda run_context: None
        self.on_eval_step_end = on_eval_step_end if on_eval_step_end else lambda run_context: None
        self.on_eval_begin = on_eval_begin if on_eval_begin else lambda run_context: None
        self.on_eval_end = on_eval_end if on_eval_end else lambda run_context: None
