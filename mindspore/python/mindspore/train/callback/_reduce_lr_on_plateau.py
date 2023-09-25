# Copyright 2022 Huawei Technologies Co., Ltd
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
"""ReduceLROnPlateau Callback class."""
from __future__ import absolute_import
from __future__ import division

import numpy as np

from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore import _checkparam as Validator
from mindspore import log as logger
from mindspore.ops import functional as F, ReduceOp
from mindspore import nn, ops
from mindspore.communication import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train.callback._callback import Callback, _handle_loss


_smaller_better_metrics = ['hausdorff_distance', 'mae', 'mse', 'loss', 'perplexity',
                           'mean_surface_distance', 'root_mean_square_distance', 'eval_loss']


class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when the monitor has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors the training
    process and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Note:
        Learning rate grouping is not supported now.

    Args:
        monitor (str): quantity to be monitored. If evaluation is performed on
            the end of train epochs, the valid monitors can be ``"loss"``,
            ``"eval_loss"`` or metric names passed when instantiate the `Model`;
            otherwise the valid monitor is ``"loss"``.
            When `monitor` is ``"loss"``, if train network has multiple outputs,
            the first element will be returned as training loss. Default: ``'eval_loss'``.
        factor (float): factor by which the learning rate will be reduced.
            `new_lr = lr * factor`. Default: ``0.1`` .
        patience (int): `monitor` value is better than history best value over
            `min_delta` is seen as improvement, `patience` is number of epochs
            with no improvement that would be waited. When the waiting
            counter `self.wait` is larger than or equal to `patience`,  the lr
            will be reduced. Default: ``10`` .
        verbose (bool): If False: quiet, if True: print related information.
            Default: ``False`` .
        mode (str): one of `{'auto', 'min', 'max'}`. In "min" mode,
            the learning rate will be reduced when the
            quantity monitored has stopped decreasing; in "max" mode it will be
            reduced when the quantity monitored has stopped increasing; in "auto"
            mode, the direction is automatically inferred from the name of the
            monitored quantity. Default: ``'auto'`` .
        min_delta (float): threshold for measuring the new optimum, to only focus on
            significant changes. Default: ``1e-4`` .
        cooldown (int): number of epochs to wait before resuming normal operation after
            lr has been reduced. Default: ``0`` .
        min_lr (float): lower bound on the learning rate. Default: ``0`` .

    Raises:
        ValueError: `mode` not in ``'auto'``, ``'min'`` or ``'max'``.
        ValueError: The monitor value is not a scalar.
        ValueError: The learning rate is not a Parameter.

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.train import Model, ReduceLROnPlateau
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics={"acc"})
        >>> # Create the dataset taking MNIST as an example. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py
        >>> dataset = create_dataset()
        >>> cb = ReduceLROnPlateau(monitor="acc", patience=3, verbose=True)
        >>> model.fit(10, dataset, callbacks=cb)
    """
    def __init__(self, monitor='eval_loss', factor=0.1, patience=10, verbose=False,
                 mode='auto', min_delta=1e-4, cooldown=0, min_lr=0):
        super(ReduceLROnPlateau, self).__init__()
        self.monitor = Validator.check_value_type('monitor', monitor, str)
        self.factor = Validator.check_float_range(factor, 0.0, 1.0, Validator.INC_NEITHER)
        self.patience = Validator.check_non_negative_int(patience)
        self.verbose = Validator.check_bool(verbose)
        self.mode = Validator.check_value_type('mode', mode, str)
        min_delta = Validator.check_value_type("min_delta", min_delta, [float, int])
        self.min_delta = abs(min_delta)
        self.cooldown = Validator.check_non_negative_int(cooldown)
        self.min_lr = Validator.check_value_type("min_lr", min_lr, [float, int])

        self.cooldown_counter = 0
        self.wait = 0
        self._reduce = ValueReduce()

        if self.mode not in ['auto', 'min', 'max']:
            raise ValueError("mode should be 'auto', 'min' or 'max', but got %s." % self.mode)
        if self.mode == 'min' or (self.mode == 'auto' and self.monitor in _smaller_better_metrics):
            self.is_improvement = lambda a, b: np.less(a, b-self.min_delta)
            self.best = np.Inf
        else:
            self.is_improvement = lambda a, b: np.greater(a, b+self.min_delta)
            self.best = -np.Inf

    def on_train_begin(self, run_context):
        """
        Initialize variables at the begin of training.

        Args:
            run_context (RunContext): Context information of the model. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        self.cooldown_counter = 0
        self.wait = 0
        if self.mode == 'min' or (self.mode == 'auto' and self.monitor in _smaller_better_metrics):
            self.best = np.Inf
        else:
            self.best = -np.Inf

    def on_train_epoch_end(self, run_context):
        """
        monitors the training process and if no improvement is seen for a 'patience' number
        of epochs, the learning rate is reduced.

        Args:
            run_context (RunContext): Context information of the model. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        cb_params = run_context.original_args()
        cur_lr = cb_params.optimizer.learning_rate
        if not isinstance(cur_lr, Parameter):
            raise ValueError("ReduceLROnPlateau does not support dynamic learning rate and group learning rate now.")

        current_monitor_value = self._get_monitor_value(cb_params)

        parallel_mode = auto_parallel_context().get_parallel_mode()
        rank_size = 1 if parallel_mode == ParallelMode.STAND_ALONE else get_group_size()
        if rank_size == 1:
            reduce_monitor_value = current_monitor_value
        else:
            reduce_monitor_value = self._reduce(Tensor(current_monitor_value, mstype.float32)).asnumpy() / rank_size

        if reduce_monitor_value is None:
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0

        if self.is_improvement(reduce_monitor_value, self.best):
            self.best = reduce_monitor_value
            self.wait = 0
        elif self.cooldown_counter <= 0:
            self.wait += 1
            if self.wait >= self.patience:
                if cur_lr > Tensor(self.min_lr):
                    new_lr = max(cur_lr * self.factor, self.min_lr)
                    F.assign(cb_params.optimizer.learning_rate, Tensor(new_lr))
                    if self.verbose:
                        print('Epoch %05d: ReduceLROnPlateau reducing learning rate to %s.'
                              % (cb_params.cur_epoch_num, new_lr))
                self.cooldown_counter = self.cooldown
                self.wait = 0

    def _get_monitor_value(self, cb_params):
        """
        Get the monitor value at the end of epoch during training.

        If `mindspore.train.callback.ReduceLROnPlateau` used with `model.train`, no evaluation process
        during training, only monitor="loss" is valid; if it used with `model.fit`, evaluation process will be
        performed at the end of epoch, valid monitor is "loss", "eval_loss" and metrics passed to `Model`.

        Args:
            cb_params (dict): A dictionary stores context information of the model. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        monitor_candidates = {}
        if self.monitor == "loss":
            loss = cb_params.get("net_outputs")
            monitor_value = _handle_loss(loss)
            if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
                logger.warning("Invalid %s.", self.monitor)
        else:
            monitor_candidates = cb_params.get("eval_results", {})
            monitor_value = monitor_candidates.get(self.monitor)

        if monitor_value is None:
            support_keys = set(["loss"] + list(monitor_candidates.keys()))
            logger.warning('Learning rate reduction is conditioned on %s, '
                           'which is not available. Available choices are: %s',
                           self.monitor, support_keys)
        if isinstance(monitor_value, np.ndarray) and monitor_value.shape != ():
            raise ValueError("ReduceLROnPlateau only supports scalar monitor now.")
        return np.array(monitor_value) if monitor_value else None


class ValueReduce(nn.Cell):
    """
    Reduces the tensor data across all devices, all devices will get the same final result.
    For more details, please refer to :class:`mindspore.ops.AllReduce`.
    """
    def __init__(self):
        super(ValueReduce, self).__init__()
        self.allreduce = ops.AllReduce(ReduceOp.SUM)

    def construct(self, x):
        return self.allreduce(x)
