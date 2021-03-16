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

from contextlib import ExitStack

from mindspore import log as logger
from mindspore.train.serialization import _fill_param_into_net
from mindspore.train.summary.summary_record import _cache_summary_tensor_data

_cur_net = None

def set_cur_net(net):
    """
    Set current net for which we are using to save checkpoint.

    Args:
        net (Cell): train network
    """
    global _cur_net
    _cur_net = net


def checkpoint_cb_for_save_op(parameter_list):
    """
    The checkpoint callback function for MindSpore.

    Will be executed by checkpoint save op.

    Args:
        parameter_list (list): Format is like [{"name",name},{"data",value}] and value type is Tensor.

    Returns:
        bool, true: means save checkpoint success.
    """
    if _cur_net is None:
        logger.warning("_cur_net is None. parameters are not updated.")
        return False

    logger.info("update parameters in the net.")
    _fill_param_into_net(_cur_net, parameter_list)
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
    Abstract base class used to build a callback class. Callbacks are context managers
    which will be entered and exited when passing into the Model.
    You can use this mechanism to initialize and release resources automatically.

    Callback function will execute some operations in the current step or epoch.

    Examples:
        >>> class Print_info(Callback):
        >>>     def step_end(self, run_context):
        >>>         cb_params = run_context.original_args()
        >>>         print(cb_params.cur_epoch_num)
        >>>         print(cb_params.cur_step_num)
        >>>
        >>> print_cb = Print_info()
        >>> model.train(epoch, dataset, callbacks=print_cb)
    """

    def __enter__(self):
        """Return the enter target."""
        return self

    def __exit__(self, *err):
        """Release resources here if have any."""

    def begin(self, run_context):
        """
        Called once before the network executing.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def epoch_begin(self, run_context):
        """
        Called before each epoch beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def epoch_end(self, run_context):
        """
        Called after each epoch finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def step_begin(self, run_context):
        """
        Called before each epoch beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def step_end(self, run_context):
        """
        Called after each step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def end(self, run_context):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """


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
                    raise TypeError("The 'callbacks' contains not-a-Callback item.")
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
        """Called once before network training."""
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
        """Called before each epoch begin."""
        for cb in self._callbacks:
            cb.step_begin(run_context)

    def step_end(self, run_context):
        """Called after each step finished."""
        for cb in self._callbacks:
            cb.step_end(run_context)

    def end(self, run_context):
        """Called once after network training."""
        for cb in self._callbacks:
            cb.end(run_context)


class InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class RunContext:
    """
    Provide information about the model.

    Provide information about original request to model function.
    Callback objects can stop the loop by calling request_stop() of run_context.

    Args:
        original_args (dict): Holding the related information of model.
    """
    def __init__(self, original_args):
        if not isinstance(original_args, dict):
            raise TypeError("The arg of RunContext should be dict type.")
        self._original_args = original_args
        self._stop_requested = False

    def original_args(self):
        """
        Get the _original_args object.

        Returns:
           Dict, an object that holds the original arguments of model.
        """
        return self._original_args

    def request_stop(self):
        """
        Set stop requirement during training.

        Callbacks can use this function to request stop of iterations.
        model.train() checks whether this is called or not.
        """
        self._stop_requested = True

    def get_stop_requested(self):
        """
        Return whether a stop is requested or not.

        Returns:
            bool, if true, model.train() stops iterations.
        """
        return self._stop_requested
