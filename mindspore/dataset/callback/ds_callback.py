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
# ==============================================================================
"""
Python callback class
"""
import threading
from mindspore._c_dataengine import PyDSCallback
from mindspore.train.callback import Callback
import mindspore.dataset as ds
from .validators import check_callback


class DSCallback:
    """
    Abstract base class used to build a dataset callback class.

    Args:
        step_size (int, optional): The number of steps before the step_begin and step_end are called (Default=1).

    Examples:
    >>> class PrintInfo(DSCallback):
    >>>     def ds_epoch_end(self, ds_run_context):
    >>>         print(cb_params.cur_epoch_num)
    >>>         print(cb_params.cur_step_num)
    >>>
    >>> data = data.map(operations=op, callbacks=PrintInfo())
    """

    @check_callback
    def __init__(self, step_size=1):
        self.step_size = step_size

    def ds_begin(self, ds_run_context):
        """
        Called before the data pipeline is started.

        Args:
            ds_run_context (RunContext): Include some information of the pipeline.
        """

    def ds_epoch_begin(self, ds_run_context):
        """
        Called before a new epoch is started.

        Args:
            ds_run_context (RunContext): Include some information of the pipeline.
        """

    def ds_epoch_end(self, ds_run_context):
        """
        Called after an epoch is finished.

        Args:
            ds_run_context (RunContext): Include some information of the pipeline.
        """

    def ds_step_begin(self, ds_run_context):
        """
        Called before n steps are started.

        Args:
            ds_run_context (RunContext): Include some information of the pipeline.
        """

    def ds_step_end(self, ds_run_context):
        """
        Called after n steps are finished.

        Args:
            ds_run_context (RunContext): Include some information of the pipeline.
        """

    def create_runtime_obj(self):
        """
        Creates a runtime (C++) object from the callback methods defined by the user.

        Returns: _c_dataengine.PyDSCallback
        """
        c_cb = PyDSCallback(self.step_size)
        at_least_one = False

        if self.__class__.ds_begin != DSCallback.ds_begin:
            c_cb.set_begin(self.ds_begin)
            at_least_one = True

        if self.__class__.ds_epoch_begin != DSCallback.ds_epoch_begin:
            c_cb.set_epoch_begin(self.ds_epoch_begin)
            at_least_one = True
        if self.__class__.ds_epoch_end != DSCallback.ds_epoch_end:
            c_cb.set_epoch_end(self.ds_epoch_end)
            at_least_one = True

        if self.__class__.ds_step_begin != DSCallback.ds_step_begin:
            c_cb.set_step_begin(self.ds_step_begin)
            at_least_one = True
        if self.__class__.ds_step_end != DSCallback.ds_step_end:
            c_cb.set_step_end(self.ds_step_end)
            at_least_one = True

        if not at_least_one:
            raise AttributeError("Provided Callback class did not override any of the 6 callback methods.")

        return c_cb


class WaitedDSCallback(Callback, DSCallback):
    """
    Abstract base class used to build a dataset callback class that are synchronized with the training callback.

    This class can be used to execute a user defined logic right after the previous step or epoch.
    For example, one augmentation needs the loss from the previous trained epoch to update some of its parameters.

    Examples:
    >>> my_cb = MyWaitedCallback(32)
    >>> data = data.map(operations=AugOp(), callbacks=my_cb)
    >>> data = data.batch(32)
    >>> # define the model
    >>> model.train(epochs, data, callbacks=[my_cb])


    Args:
       step_size: the number of rows in each step.
       Usually the step size will be equal to the batch size (Default=1)
    """

    def __init__(self, step_size=1):
        super().__init__()
        self.step_size = step_size
        self.step_event = threading.Event()
        self.step_run_context = None

        self.epoch_event = threading.Event()
        self.epoch_run_context = None

        self.training_ended = False

    def sync_epoch_begin(self, train_run_context, ds_run_context):
        """
        Called before a new dataset epoch is started and after the previous training epoch is ended.

        Args:
            train_run_context: Include some information of the model with feedback from the previous epoch.
            ds_run_context: Include some information of the dataset pipeline.
        """

    def sync_step_begin(self, train_run_context, ds_run_context):
        """
        Called before a new dataset step is started and after the previous training step is ended.

        Args:
            train_run_context: Include some information of the model with feedback from the previous step.
            ds_run_context: Include some information of the dataset pipeline.
        """

    def epoch_end(self, run_context):
        """
        Internal method, do not call/override. Defines epoch_end of Callback to release the wait in ds_epoch_begin.

        Args:
          run_context: Include some information of the model.
        """
        self.epoch_run_context = run_context
        self.epoch_event.set()

    def ds_epoch_begin(self, ds_run_context):
        """
        Internal method, do not call/override. Defines ds_epoch_begin of DSCallback to wait for MS epoch_end callback.

        Args:
          ds_run_context: Include some information of the pipeline.
        """
        if ds_run_context.cur_epoch_num > 1:
            if not self.training_ended:
                success = self.epoch_event.wait(timeout=ds.config.get_callback_timeout())
                self.epoch_event.clear()
                if not success:
                    raise RuntimeError(f"ds_epoch_begin timed out after {ds.config.get_callback_timeout()} second(s).")
            # by the time this thread wakes up, self.epoch_run_context is already available
            self.sync_epoch_begin(self.epoch_run_context, ds_run_context)

    def step_end(self, run_context):
        """
        Internal method, do not call/override. Defines step_end of Callback to release the wait in ds_step_begin.

        Args:
          run_context: Include some information of the model.
        """
        self.step_run_context = run_context
        self.step_event.set()

    def ds_step_begin(self, ds_run_context):
        """
        Internal method, do not call/override. Defines ds_step_begin of DSCallback to wait for MS step_end callback.

        Args:
            ds_run_context: Include some information of the pipeline.
        """
        if ds_run_context.cur_step_num > self.step_size:
            if not self.training_ended:
                success = self.step_event.wait(timeout=ds.config.get_callback_timeout())
                self.step_event.clear()
                if not success:
                    raise RuntimeError(f"ds_step_begin timed out after {ds.config.get_callback_timeout()} second(s).")
                # by the time this thread wakes up, self.epoch_run_context is already available
            self.sync_step_begin(self.step_run_context, ds_run_context)

    def create_runtime_obj(self):
        """
        Creates a runtime (C++) object from the callback methods defined by the user. This method is internal.

        Returns: _c_dataengine.PyDSCallback
        """
        c_cb = PyDSCallback(self.step_size)
        at_least_one = False

        if self.__class__.sync_step_begin != WaitedDSCallback.sync_step_begin:
            c_cb.set_step_begin(self.ds_step_begin)
            at_least_one = True

        if self.__class__.sync_epoch_begin != WaitedDSCallback.sync_epoch_begin:
            c_cb.set_epoch_begin(self.ds_epoch_begin)
            at_least_one = True

        if not at_least_one:
            raise AttributeError("Provided Callback class did not override any of the 2 callback methods.")

        return c_cb

    def end(self, run_context):
        self.epoch_end(run_context)
        self.step_end(run_context)
        self.training_ended = True
