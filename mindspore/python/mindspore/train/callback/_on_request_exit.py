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
"""OnRequestExit Callback class."""

from __future__ import absolute_import
import os
import signal

from mindspore import log
from mindspore._checkparam import Validator
from mindspore.train.serialization import load_checkpoint, save_checkpoint, export
from mindspore.train.callback._callback import Callback


class OnRequestExit(Callback):
    """
    Respond to the user's closing request, exit the training or eval process, and save the checkpoint and mindir.

    Register OnRequestExit Callback before training, when the user want to exit the training process
    and save the training data, could send the registered exit signal 'sig' to the training process.
    After the training process executes the current step, saves the current training status,
    including checkpoint and mindir, and then exit the training process.

    Args:
        save_ckpt (bool): Whether save the checkpoint before the training process exit. Default: True.
        save_mindir (bool): Whether save the mindir before the training process exit. Default: True.
        file_name (str): The saved checkpoint and mindir file name,
            the checkpoint file add suffix '.ckpt', the mindir file add suffix '.mindir'. Default: 'Net'.
        directory (str): The directory save checkpoint and mindir. Default: './'.
        sig (int): The user registered exit signal, it must be a captureable and negligible signal.
            When the process receives the signal, exits the training or eval process. Default: signal.SIGTERM.

    Raises:
        ValueError: If the 'save_ckpt' is not a bool.
        ValueError: If the 'save_mindir' is not a bool.
        ValueError: If the 'file_name' is not a str.
        ValueError: If the 'directory' is not a str.
        ValueError: If the 'sig' is not an int or the 'sig' is signal.SIGKILL.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import dataset as ds
        >>> from mindspore import nn
        >>>
        >>> # Define the forward net
        >>> class ForwardNet(nn.Cell):
        >>>     def __init__(self, num_class=10, channel=1):
        >>>         super(ForwardNet, self).__init__()
        >>>         self.param = ms.Parameter(1.0)
        >>>         self.relu = ms.ops.ReLU()
        >>>
        >>>     def construct(self, x):
        >>>         return self.relu(x + self.param)
        >>> forward_net = ForwardNet()
        >>> loss = nn.MAELoss()
        >>> opt = nn.Momentum(forward_net.trainable_params(), 0.01, 0.9)
        >>> model = ms.Model(forward_net, loss_fn=loss, optimizer=opt)\
        >>>
        >>> # Create dataset
        >>> def generator_multi_column():
        >>>    i = 0
        >>>    while i < 1000:
        >>>        i += 1
        >>>        yield np.ones((1, 32, 32)).astype(np.float32) * 0.01, np.array(1).astype(np.int32)
        >>> dataset = ds.GeneratorDataset(source=generator_multi_column, column_names=["data", "label"])
        >>> dataset = dataset.batch(32, drop_remainder=True)
        >>>
        >>> on_request_exit = ms.train.OnRequestExit(file_name='LeNet5')
        >>> model.train(10, dataset, callbacks=on_request_exit)
        >>> # The user send the signal SIGTERM to the training process,
        >>> # the process would save the checkpoint and mindir, and then exit the training process.
    """

    def __init__(self, save_ckpt=True, save_mindir=True, file_name='Net', directory='./', sig=signal.SIGTERM):
        super(OnRequestExit, self).__init__()
        self.save_ckpt = Validator.check_isinstance('save_ckpt', save_ckpt, bool)
        self.save_mindir = Validator.check_isinstance('save_mindir', save_mindir, bool)
        if self.save_ckpt or self.save_mindir:
            file_name = Validator.check_isinstance('file_name', file_name, str)
            directory = Validator.check_isinstance('directory', directory, str)
            os.makedirs(os.path.abspath(directory), exist_ok=True)
            self.train_file_path = os.path.abspath(os.path.join(directory, f"{file_name}_train"))
            self.eval_file_path = os.path.abspath(os.path.join(directory, f"{file_name}_eval"))
        self.sig = Validator.check_isinstance('sig', sig, int)
        if hasattr(signal, "SIGKILL") and self.sig == signal.SIGKILL:
            raise ValueError("Not support send exit request by signal SIGKILL.")
        self.exit = False

    def on_train_begin(self, run_context):
        """
        When the train begin, register the handler for exit signal transferred by user.

        Args:
            run_context (RunContext): Context information of the model.
                For more details, please refer to :class:`mindspore.train.RunContext`.
        """
        signal.signal(self.sig, self._handle_signal)
        if self.save_ckpt and os.path.isfile(f"{self.train_file_path}.ckpt"):
            cb_params = run_context.original_args()
            train_net = cb_params.train_network
            load_checkpoint(f"{self.train_file_path}.ckpt", net=train_net)

    def on_train_step_end(self, run_context):
        """
        When the train step end, if received the exit signal, set the 'run_context' attribute '_stop_requested' to True.
        Then exit the training process after this step training.

        Args:
            run_context (RunContext): Include some information of the model.
                For more details, please refer to :class:`mindspore.train.RunContext`.
        """
        if self.exit:
            run_context.request_stop()

    def on_train_epoch_end(self, run_context):
        """
        When the train epoch end, if received the exit signal,
        set the 'run_context' attribute '_stop_requested' to True.
        Then exit the training process after this epoch training.

        Args:
            run_context (RunContext): Include some information of the model.
                For more details, please refer to :class:`mindspore.train.RunContext`.
        """
        if self.exit:
            run_context.request_stop()

    def on_train_end(self, run_context):
        """
        When the train end, if received the exit signal,
        the checkpoint and mindir would be saved according to the user config.

        Args:
            run_context (RunContext): Include some information of the model.
                For more details, please refer to :class:`mindspore.train.RunContext`.
        """
        if not self.exit:
            return
        cb_params = run_context.original_args()
        train_net = cb_params.train_network
        if self.save_ckpt:
            save_checkpoint(train_net, ckpt_file_name=self.train_file_path)
        if self.save_mindir:
            inputs = cb_params.train_dataset_element
            export(train_net, *inputs, file_name=self.train_file_path, file_format='MINDIR')

    def on_eval_begin(self, run_context):
        """
        When the eval begin, register the handler for exit signal transferred by user.

        Args:
            run_context (RunContext): Context information of the model.
                For more details, please refer to :class:`mindspore.train.RunContext`.
        """
        signal.signal(self.sig, self._handle_signal)
        if not self.save_ckpt:
            return
        cb_params = run_context.original_args()
        eval_net = cb_params.eval_network
        if os.path.isfile(f"{self.eval_file_path}.ckpt"):
            load_checkpoint(f"{self.eval_file_path}.ckpt", net=eval_net)
        elif os.path.isfile(f"{self.train_file_path}.ckpt"):
            load_checkpoint(f"{self.train_file_path}.ckpt", net=eval_net)

    def on_eval_step_end(self, run_context):
        """
        When the eval step end, if received the exit signal, set the 'run_context' attribute '_stop_requested' to True.
        Then exit the eval process after this step eval.

        Args:
            run_context (RunContext): Include some information of the model.
                For more details, please refer to :class:`mindspore.train.RunContext`.
        """
        if self.exit:
            run_context.request_stop()

    def on_eval_end(self, run_context):
        """
        When the eval end, if received the exit signal,
        the checkpoint and mindir would be saved according to the user config.

        Args:
            run_context (RunContext): Include some information of the model.
                For more details, please refer to :class:`mindspore.train.RunContext`.
        """
        if not self.exit:
            return
        cb_params = run_context.original_args()
        eval_net = cb_params.eval_network
        if self.save_ckpt:
            save_checkpoint(eval_net, ckpt_file_name=self.eval_file_path)
        if self.save_mindir:
            inputs = cb_params.eval_dataset_element
            export(eval_net, *inputs, file_name=self.eval_file_path, file_format='MINDIR')

    def _handle_signal(self, signum, frame):
        """Handle the received signal"""
        log.debug(f"signum: {signum}, frame: {frame}")
        self.exit = True
