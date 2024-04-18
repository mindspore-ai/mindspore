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

import copy
import numpy as np

from mindspore import ops, nn
from mindspore.common.tensor import Tensor
from mindspore import _checkparam as Validator
from mindspore.train.serialization import load_param_into_net
from mindspore import log as logger
from mindspore.ops import ReduceOp
from mindspore.communication import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train.callback._callback import Callback, _handle_loss


_smaller_better_metrics = ['hausdorff_distance', 'mae', 'mse', 'loss', 'perplexity',
                           'mean_surface_distance', 'root_mean_square_distance', 'eval_loss']


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.

    Assuming `monitor` is "accuracy", with this, `mode` would be "max" since
    goal of trianing is to maximize the accuracy, the `model.fit()` training
    loop will check at end of epoch whether the accuracy is no longer
    increasing, considering the `min_delta` and `patience` if applicable.
    Once it's found no longer increasing, `run_context.request_stop()`
    will be called and the training terminates.

    Args:
        monitor (str): quantity to be monitored. If evaluation is performed on
            the end of train epochs, the valid monitors can be "loss",
            "eval_loss" or metric names passed when instantiate the `Model`;
            otherwise the valid monitor is "loss".
            When monitor is "loss", if train network has multiple outputs,
            the first element will be returned as training loss.
            Default: ``'eval_loss'`` .
        patience (int): `monitor` value is better than history best value over
            `min_delta` is seen as improvement, `patience` is number of epochs
            with no improvement that would be waited. When the waiting
            counter `self.wait` is larger than or equal to `patience`,  the
            training process will be stopped. Default: ``0`` .
        verbose (bool): If False: quiet, if True: print related information.
            Default: ``False`` .
        mode (str): one of `{'auto', 'min', 'max'}`. In "min" mode,
            the learning rate will be reduced when the
            quantity monitored has stopped decreasing; in "max" mode it will be
            reduced when the quantity monitored has stopped increasing; in "auto"
            mode, the direction is automatically inferred from the name of the
            monitored quantity. Default: ``'auto'`` .
        min_delta (float): threshold for measuring the new optimum, to only focus on
            significant changes. Default: ``0`` .
        baseline (float): Baseline value for the monitor. When the monitor value shows
            improvement over the history best value and the baseline, the internal
            wait counter will be set to zero. Default: ``None`` .
        restore_best_weights (bool): Whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used. Default: ``False`` .

    Raises:
        ValueError: `mode` not in 'auto', 'min' or 'max'.
        ValueError: The monitor value is not a scalar.

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.train import Model, EarlyStopping
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics={"acc"})
        >>> # Create the dataset taking MNIST as an example. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py
        >>> dataset = create_dataset()
        >>> cb = EarlyStopping(monitor="acc", patience=3, verbose=True)
        >>> model.fit(10, dataset, callbacks=cb)
    """

    def __init__(self, monitor='eval_loss', min_delta=0, patience=0,
                 verbose=False, mode='auto', baseline=None, restore_best_weights=False):
        super(EarlyStopping, self).__init__()
        self.monitor = Validator.check_value_type('monitor', monitor, str)
        min_delta = Validator.check_value_type("min_delta", min_delta, [float, int])
        self.min_delta = abs(min_delta)
        self.patience = Validator.check_non_negative_int(patience)
        self.verbose = Validator.check_bool(verbose)
        self.mode = Validator.check_value_type('mode', mode, str)
        self.baseline = Validator.check_value_type("min_delta", min_delta, [float, int]) if baseline else None
        self.restore_best_weights = Validator.check_bool(restore_best_weights)

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights_param_dict = None
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

        self.wait = 0
        self.stopped_epoch = 0
        if self.mode == 'min' or (self.mode == 'auto' and self.monitor in _smaller_better_metrics):
            self.best = np.Inf
        else:
            self.best = -np.Inf
        self.best_weights_param_dict = None

    def on_train_epoch_end(self, run_context):
        """
        monitors the training process and if no improvement is seen for a 'patience' number
        of epochs, the training process will be stopped.

        Args:
            run_context (RunContext): Context information of the model. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        cb_params = run_context.original_args()

        cur_epoch = cb_params.get("cur_epoch_num")
        current_value = self._get_monitor_value(cb_params)

        parallel_mode = auto_parallel_context().get_parallel_mode()
        rank_size = 1 if parallel_mode == ParallelMode.STAND_ALONE else get_group_size()
        if rank_size == 1:
            current = current_value
        else:
            current = self._reduce(Tensor(current_value.astype(np.float32))) / rank_size

        if current is None:
            return

        if self.restore_best_weights and self.best_weights_param_dict is None:
            self.best_weights_param_dict = copy.deepcopy(cb_params.train_network.parameters_dict())
        self.wait += 1
        if self.is_improvement(current, self.best):
            self.best = current
            if self.restore_best_weights:
                self.best_weights_param_dict = copy.deepcopy(cb_params.train_network.parameters_dict())
            if self.baseline is None or self.is_improvement(current, self.baseline):
                self.wait = 0

        if self.wait >= self.patience:
            self.stopped_epoch = cur_epoch
            run_context.request_stop()
            if self.restore_best_weights and self.best_weights_param_dict is not None:
                if self.verbose:
                    print('Restoring model weights from the end of the best epoch.')
                load_param_into_net(cb_params.train_network, self.best_weights_param_dict)

    def on_train_end(self, run_context):
        """
        If verbose is True, print the stopped epoch.

        Args:
            run_context (RunContext): Context information of the model. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """

        if self.stopped_epoch > 0 and self.verbose:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))

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
            logger.warning('Early stopping is conditioned on %s, '
                           'which is not available. Available choices are: %s',
                           self.monitor, support_keys)
        if isinstance(monitor_value, np.ndarray) and monitor_value.shape != ():
            raise ValueError("EarlyStopping only supports scalar monitor now.")
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
        return self.allreduce(x).asnumpy()
