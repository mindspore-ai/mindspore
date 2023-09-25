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
"""Callback related classes and functions."""
from __future__ import absolute_import

from contextlib import ExitStack
import numpy as np

from mindspore.common.tensor import Tensor
from mindspore import log as logger
from mindspore.train.summary.summary_record import _cache_summary_tensor_data
from mindspore.common.parameter import Parameter
from mindspore.train.serialization import load_param_into_net
from mindspore.common import dtype as mstype

CUR_NET = None


def set_cur_net(net):

    """
    Set current net for which we are using to save checkpoint.

    Args:
        net (Cell): train network
    """
    global CUR_NET
    CUR_NET = net


def _fill_param_into_net(net, parameter_list):
    """
    Fills parameter_list into net.

    Args:
        net (Cell): train network.
        parameter_list (list): parameters list from ge callback.
    """
    parameter_dict = {}
    while parameter_list:
        tmp_param = parameter_list.pop(0)
        param_name = tmp_param["name"]
        param_data = tmp_param["data"]
        if isinstance(param_data, Parameter):
            param_data.init_data()
        np_val = param_data.asnumpy()

        if np_val.shape == (1,):
            parameter_dict[param_name] = Parameter(np_val, name=param_name)
        elif np_val.shape == ():
            parameter_dict[param_name] = Parameter(Tensor(np_val.tolist(), mstype.pytype_to_dtype(np_val.dtype)),
                                                   name=param_name)
        else:
            parameter_dict[param_name] = Parameter(Tensor(np_val), name=param_name)

    load_param_into_net(net, parameter_dict, strict_load=True)


def checkpoint_cb_for_save_op(parameter_list):
    """
    The checkpoint callback function for MindSpore.

    Will be executed by checkpoint save op.

    Args:
        parameter_list (list): Format is like [{"name",name},{"data",value}] and value type is Tensor.

    Returns:
        bool, true: means save checkpoint success.
    """
    if CUR_NET is None:
        logger.warning("CUR_NET is None. parameters are not updated.")
        return False

    logger.info("update parameters in the net.")
    _fill_param_into_net(CUR_NET, parameter_list)
    set_cur_net(None)
    return True


def summary_cb_for_save_op(summary_list):
    """
    The summary callback function for MindSpore.

    Will be executed by summary op.

    Args:
        summary_list (list): Format is like [{"name": tag_name, "data": tensor},...] and value is Scalar/Tensor.

    Returns:
        bool, true: means save summary success.
    """
    ret = _cache_summary_tensor_data(summary_list)
    return ret


class Callback:
    """
    Abstract base class used to build a Callback class. Callbacks are context managers
    which will be entered and exited when passing into the Model.
    You can use this mechanism to do some custom operations.

    Each method of Callback class corresponds to a stage in training or eval process, and those methods
    have the same input `run_context`, which hold context information of the model in
    training or eval process. When defining a Callback subclass or creating a custom Callback,
    note that you should override methods with names prefixed with "on_train" or "on_eval",
    otherwise ValueError will be raised if the custimized Callbacks used in `model.fit`.

    When creating a custom Callback, model context information can be obtained in Callback
    methods by calling `RunContext.original_args()`, which is a dictionary varivable
    recording current attributes. Users can add custimized attributes to the information.
    Training process can also be stopped by calling `request_stop` method. For details
    of custom Callback, please check
    `Callback tutorial <https://www.mindspore.cn/tutorials/en/master/advanced/model/
    callback.html#customized-callback-mechanism>`_.

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn
        >>> from mindspore import dataset as ds
        >>> from mindspore.train import Model, Callback
        >>> class Print_info(Callback):
        ...     def step_end(self, run_context):
        ...         cb_params = run_context.original_args()
        ...         print("step_num: ", cb_params.cur_step_num)
        >>>
        >>> print_cb = Print_info()
        >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
        >>> dataset = ds.NumpySlicesDataset(data=data).batch(32)
        >>> net = nn.Dense(10, 5)
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
        >>> model.train(1, dataset, callbacks=print_cb)
        step_num: 1
        step_num: 2
    """

    def __enter__(self):
        """Return the enter target."""
        return self

    def __exit__(self, *err):
        """Release resources here if have any."""

    def begin(self, run_context):
        """
        Called once before the network executing.
        A backwards compatibility alias for `on_train_begin` and `on_eval_begin`.

        Note:
            `begin` is deprecated and will be deleted in a future version,
            please use `on_train_begin` and `on_eval_begin` instead.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def epoch_begin(self, run_context):
        """
        Called before each epoch beginning.
        A backwards compatibility alias for `on_train_epoch_begin` and `on_eval_epoch_begin`.

        Note:
            `epoch_begin` is deprecated and will be deleted in a future version,
            please use `on_train_epoch_begin` and `on_eval_epoch_begin` instead.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def epoch_end(self, run_context):
        """
        Called after each epoch finished.
        A backwards compatibility alias for `on_train_epoch_end` and `on_eval_epoch_end`.

        Note:
            `epoch_end` is deprecated and will be deleted in a future version,
            please use `on_train_epoch_end` and `on_eval_epoch_end` instead.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def step_begin(self, run_context):
        """
        Called before each step beginning.
        A backwards compatibility alias for `on_train_step_begin` and `on_eval_step_begin`.

        Note:
            `step_begin` is deprecated and will be deleted in a future version,
            please use `on_train_step_begin` and `on_eval_step_begin` instead.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def step_end(self, run_context):
        """
        Called after each step finished.
        A backwards compatibility alias for `on_train_step_end` and `on_eval_step_end`.

        Note:
            `step_end` is deprecated and will be deleted in a future version,
            please use `on_train_step_end` and `on_eval_step_end` instead.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def end(self, run_context):
        """
        Called once after network training.
        A backwards compatibility alias for `on_train_end` and `on_eval_end`.

        Note:
            `end` is deprecated and will be deleted in a future version,
            please use `on_train_end` and `on_eval_end` instead.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def on_train_begin(self, run_context):
        """
        Called once before the network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.begin(run_context)

    def on_train_epoch_begin(self, run_context):
        """
        Called before each training epoch begin.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_begin(run_context)

    def on_train_epoch_end(self, run_context):
        """
        Called after each training epoch end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_end(run_context)

    def on_train_step_begin(self, run_context):
        """
        Called before each training step begin.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.step_begin(run_context)

    def on_train_step_end(self, run_context):
        """
        Called after each training step end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.step_end(run_context)

    def on_train_end(self, run_context):
        """
        Called after training end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.end(run_context)

    def on_eval_begin(self, run_context):
        """
        Called before eval begin.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.begin(run_context)

    def on_eval_epoch_begin(self, run_context):
        """
        Called before eval epoch begin.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_begin(run_context)

    def on_eval_epoch_end(self, run_context):
        """
        Called after eval epoch end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_end(run_context)

    def on_eval_step_begin(self, run_context):
        """
        Called before each eval step begin.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.step_begin(run_context)

    def on_eval_step_end(self, run_context):
        """
        Called after each eval step end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.step_end(run_context)

    def on_eval_end(self, run_context):
        """
        Called after eval end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.end(run_context)


class CallbackManager(Callback):
    """
    Sequential execution of callback functions.

    Execute Callback functions at certain points.

    Args:
        callbacks (Optional[list[Callback], Callback]): None, callback, or callbacks list.
    """

    def __init__(self, callbacks):
        self._callbacks, self._stack = [], None
        if isinstance(callbacks, Callback):
            self._callbacks.append(callbacks)
        elif isinstance(callbacks, list):
            for cb in callbacks:
                if not isinstance(cb, Callback):
                    raise TypeError("When the 'callbacks' is a list, the elements in "
                                    "'callbacks' must be Callback functions.")
                self._callbacks.append(cb)
        elif callbacks is not None:
            raise TypeError("The 'callbacks' is not a Callback or a list of Callback.")

    def __enter__(self):
        if self._stack is None:
            callbacks, self._stack = [], ExitStack().__enter__()
            for callback in self._callbacks:
                target = self._stack.enter_context(callback)
                if not isinstance(target, Callback):
                    logger.warning("Please return 'self' or a Callback as the enter target.")
                    callbacks.append(callback)
                else:
                    callbacks.append(target)
            self._callbacks = callbacks
        return self

    def __exit__(self, *err):
        return self._stack.__exit__(*err)

    def begin(self, run_context):
        """Called once before network train or eval."""
        for cb in self._callbacks:
            cb.begin(run_context)

    def epoch_begin(self, run_context):
        """Called before each epoch begin."""
        for cb in self._callbacks:
            cb.epoch_begin(run_context)

    def epoch_end(self, run_context):
        """Called after each epoch finished."""
        for cb in self._callbacks:
            cb.epoch_end(run_context)

    def step_begin(self, run_context):
        """Called before each step begin."""
        for cb in self._callbacks:
            cb.step_begin(run_context)

    def step_end(self, run_context):
        """Called after each step finished."""
        for cb in self._callbacks:
            cb.step_end(run_context)

    def end(self, run_context):
        """Called once after network train or eval."""
        for cb in self._callbacks:
            cb.end(run_context)

    def on_train_begin(self, run_context):
        """Called before network train."""
        for cb in self._callbacks:
            cb.on_train_begin(run_context)

    def on_train_epoch_begin(self, run_context):
        """Called before each train epoch begin."""
        for cb in self._callbacks:
            cb.on_train_epoch_begin(run_context)

    def on_train_epoch_end(self, run_context):
        """Called after each train epoch finished."""
        for cb in self._callbacks:
            cb.on_train_epoch_end(run_context)

    def on_train_step_begin(self, run_context):
        """Called before each train step begin."""
        for cb in self._callbacks:
            cb.on_train_step_begin(run_context)

    def on_train_step_end(self, run_context):
        """Called after each train step finished."""
        for cb in self._callbacks:
            cb.on_train_step_end(run_context)

    def on_train_end(self, run_context):
        """Called after network train end."""
        for cb in self._callbacks:
            cb.on_train_end(run_context)

    def on_eval_begin(self, run_context):
        """Called before network eval."""
        for cb in self._callbacks:
            cb.on_eval_begin(run_context)

    def on_eval_epoch_begin(self, run_context):
        """Called before eval epoch begin."""
        for cb in self._callbacks:
            cb.on_eval_epoch_begin(run_context)

    def on_eval_epoch_end(self, run_context):
        """Called after eval epoch finished."""
        for cb in self._callbacks:
            cb.on_eval_epoch_end(run_context)

    def on_eval_step_begin(self, run_context):
        """Called before each eval step begin."""
        for cb in self._callbacks:
            cb.on_eval_step_begin(run_context)

    def on_eval_step_end(self, run_context):
        """Called after each eval step finished."""
        for cb in self._callbacks:
            cb.on_eval_step_end(run_context)

    def on_eval_end(self, run_context):
        """Called after network eval end."""
        for cb in self._callbacks:
            cb.on_eval_end(run_context)


class InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class RunContext:
    """
    Hold and manage information about the model.

    `RunContext` is mainly used to collect context-related information about the model during
    training or eval and pass it into the Callback object as an input parameter to share information.

    Callback objects not only can obtain the Model context information by calling by
    `RunContext.original_args()` and add extra attributes to the information, but also can stop the
    training process by calling `request_stop` method. For details of custom Callback,
    please check
    `Callback Mechanism <https://www.mindspore.cn/tutorials/en/master/advanced/model/callback.html>`_.

    `RunContext.original_args()` holds the model context information as a dictionary variable, and
    different attributes of the dictionary are stored in training or eval process. Details are as follows:

    +--------------------------------+-------------------------------+---------------------------------------+
    |  Attributes supported in train |  Attributes supported in eval |               meaning                 |
    +================================+===============================+=======================================+
    |      train_network             |                               | train network with optimizer and loss |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      epoch_num                 |                               |      Number of train epochs           |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      train_dataset             |                               |         the train dataset             |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      loss_fn                   |                               |         the loss function             |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      optimizer                 |                               |         the optimizer                 |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      parallel_mode             |                               |         the parallel mode             |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      device_number             |                               |         the device number             |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      train_dataset_element     |                               | the train data element of current step|
    +--------------------------------+-------------------------------+---------------------------------------+
    |      last_save_ckpt_step       |                               |   the last step num of save ckpt      |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      latest_ckpt_file          |                               |         the ckpt file                 |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      cur_epoch_num             |                               |         number of current epoch       |
    +--------------------------------+-------------------------------+---------------------------------------+
    |                                |       eval_network            |     the evaluate network              |
    +--------------------------------+-------------------------------+---------------------------------------+
    |                                |       valid_dataset           |     the valid dataset                 |
    +--------------------------------+-------------------------------+---------------------------------------+
    |                                |       metrics                 |     the evaluate metrics              |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      mode                      |       mode                    |     "train" or "eval"                 |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      batch_num                 |       batch_num               |    the train/eval batch number        |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      list_callback             |       list_callback           |       callback list                   |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      network                   |       network                 |       basic network                   |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      cur_step_num              |       cur_step_num            |    the train/eval step number         |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      dataset_sink_mode         |       dataset_sink_mode       |    the train/eval sink mode           |
    +--------------------------------+-------------------------------+---------------------------------------+
    |      net_outputs               |       net_outputs             |     network output results            |
    +--------------------------------+-------------------------------+---------------------------------------+

    Args:
        original_args (dict): Holding the related information of model.

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore.train import RunContext
        >>> cb_params = {}
        >>> cb_params["cur_epoch_num"] = 4
        >>> cb_params["epoch_num"] = 4
        >>> cb_params["cur_step_num"] = 2
        >>> cb_params["batch_num"] = 2
        >>> cb_params["net_outputs"] = Tensor(2.0)
        >>> run_context = RunContext(cb_params)
        >>> whether_stop = run_context.get_stop_requested()
    """
    def __init__(self, original_args):
        if not isinstance(original_args, dict):
            raise TypeError("The argument 'original_args' of RunContext should be dict type, "
                            "but got {}.".format(type(original_args)))
        self._original_args = original_args
        self._stop_requested = False

    def original_args(self):
        """
        Get the _original_args object.

        Returns:
           Dict, an object that holds the original arguments of model.

        Tutorial Examples:
            - `Callback Mechanism - Customized Callback Mechanism
              <https://mindspore.cn/tutorials/en/master/advanced/model/callback.html#customized-callback-mechanism>`_
        """
        return self._original_args

    def request_stop(self):
        """
        Set stop requirement during training or eval.

        Callbacks can use this function to request stop of iterations.
        model.train() checks whether this is called or not.

        Tutorial Examples:
            - `Callback Mechanism - Customized Training Termination Time
              <https://mindspore.cn/tutorials/en/master/advanced/model/callback.html#
              customized-training-termination-time>`_
        """
        self._stop_requested = True

    def get_stop_requested(self):
        """
        Return whether a stop is requested or not.

        Returns:
            bool, if true, model.train() stops iterations.
        """
        return self._stop_requested


def _handle_loss(loss):
    """Handle loss."""
    if isinstance(loss, (tuple, list)):
        if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
            loss = loss[0]
    elif isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
        loss = float(np.mean(loss.asnumpy()))
    return loss
