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
"""LearningRateScheduler Callback class."""
from __future__ import absolute_import

import math
import numpy as np

from mindspore import log as logger
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.train.callback._callback import Callback
from mindspore.ops import functional as F


class LearningRateScheduler(Callback):
    """
    Change the learning_rate during training.

    Args:
        learning_rate_function (Function): The function about how to change the learning rate during training.

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn
        >>> from mindspore.train import Model, LearningRateScheduler
        >>> from mindspore import dataset as ds
        ...
        >>> def learning_rate_function(lr, cur_step_num):
        ...     if cur_step_num%1000 == 0:
        ...         lr = lr*0.1
        ...     return lr
        ...
        >>> lr = 0.1
        >>> momentum = 0.9
        >>> net = nn.Dense(10, 5)
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> optim = nn.Momentum(net.trainable_params(), learning_rate=lr, momentum=momentum)
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
        ...
        >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
        >>> dataset = ds.NumpySlicesDataset(data=data).batch(32)
        >>> model.train(1, dataset, callbacks=[LearningRateScheduler(learning_rate_function)],
        ...             dataset_sink_mode=False)
    """

    def __init__(self, learning_rate_function):
        super(LearningRateScheduler, self).__init__()
        self.learning_rate_function = learning_rate_function

    def step_end(self, run_context):
        """
        Change the learning_rate at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        arr_lr = cb_params.optimizer.learning_rate.asnumpy()
        lr = float(np.array2string(arr_lr))
        new_lr = self.learning_rate_function(lr, cb_params.cur_step_num)
        if not math.isclose(lr, new_lr, rel_tol=1e-10):
            F.assign(cb_params.optimizer.learning_rate, Tensor(new_lr, mstype.float32))
            logger.info(f'At step {cb_params.cur_step_num}, learning_rate change to {new_lr}')
