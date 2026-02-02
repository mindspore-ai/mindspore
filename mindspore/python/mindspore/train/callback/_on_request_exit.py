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
import json
import signal
import threading
from mindspore.common import dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.train._utils import _make_directory
from mindspore import _checkparam as Validator
from mindspore.train.serialization import load_checkpoint, save_checkpoint, export
from mindspore.communication.management import get_group_size
from mindspore.train.callback._callback import Callback
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.context import ParallelMode


class OnRequestExit(Callback):
    """
    Respond to the user's closing request, exit the training or eval process, and save the checkpoint and mindir.

    Register OnRequestExit Callback before training, when the user wants to exit the training process
    and save the training data, could send the registered exit signal 'sig' to the training process or modify the
    'GracefulExit' that is a key in the JSON file specified by the 'config_file' to '1'.
    After the training process executes the current step, saves the current training status,
    including checkpoint and mindir, and then exit the training process.

    Args:
        save_ckpt (bool): Whether to save the checkpoint before the training process exit. Default: ``True`` .
        save_mindir (bool): Whether to save the mindir before the training process exit. Default: ``True`` .
        file_name (str): The saved checkpoint and mindir file name,
            the checkpoint file add suffix '.ckpt', the mindir file add suffix '.mindir'. Default: ``'Net'`` .
        directory (str): The path to save files. It will generate a 'rank_{id}' path by rank_id
            to save checkpoint and mindir. Default: ``'./'`` .
        sig (int): The user registered exit signal, it must be a capturable and negligible signal.
            When the process receives the signal, exits the training or eval process. Default: ``signal.SIGTERM`` .
        config_file (str): A json config file used to exit training process gracefully. Key: ``{"GracefulExit": 1}`` .
            Default: ``None`` .

    Raises:
        ValueError: If the 'save_ckpt' is not a bool.
        ValueError: If the 'save_mindir' is not a bool.
        ValueError: If the 'file_name' is not a str.
        ValueError: If the 'directory' is not a str.
        ValueError: If the 'sig' is not an int or the 'sig' is not ``signal.SIGTERM``.

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.train import Model, TimeMonitor
        >>> import mindspore as ms
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
        >>> # Create the dataset taking MNIST as an example. Refer to
        >>> # https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py
        >>> dataset = create_dataset()
        >>> on_request_exit = ms.train.OnRequestExit(file_name='LeNet5')
        >>> model.train(10, dataset, callbacks=on_request_exit)
    """

    def __init__(self, save_ckpt=True, save_mindir=True, file_name='Net', directory='./', config_file=None,
                 sig=signal.SIGTERM):
        super(OnRequestExit, self).__init__()
        self.save_ckpt = Validator.check_isinstance('save_ckpt', save_ckpt, bool)
        self.save_mindir = Validator.check_isinstance('save_mindir', save_mindir, bool)
        self.sig = Validator.check_isinstance('sig', sig, int)
        if hasattr(signal, "SIGKILL") and self.sig == signal.SIGKILL:
            raise ValueError("Not support send exit request by signal SIGKILL.")
        self.exit = False  # used signal to exit the training process
        self.lock = threading.Lock()
        self.save_path = directory
        self.key = "GracefulExit"
        self.remote_config_file = config_file  # used config file to save checkpoint and exit training process
        self.use_graceful = os.environ.get("MS_ENABLE_GRACEFUL_EXIT") == "1"
        self.is_distributed = get_group_size() > 1
        self.integrated_save = True
        self.stop_train = False
        self.need_do_step_end = False
        if self.save_ckpt or self.save_mindir:
            self.train_name, self.eval_name = self._get_save_path(file_name)

    def on_train_begin(self, run_context):
        """
        When the train begin, register the handler for exit signal transferred by user.

        Args:
            run_context (RunContext): Context information of the model.
                For more details, please refer to :class:`mindspore.train.RunContext`.
        """
        signal.signal(self.sig, self._handle_signal)
        if self.save_ckpt and os.path.isfile(f"{self.train_name}.ckpt"):
            cb_params = run_context.original_args()
            train_net = cb_params.train_network
            load_checkpoint(f"{self.train_name}.ckpt", net=train_net)

    def on_train_step_begin(self, run_context):
        """
        Check whether received the exit signal or
        whether the value of 'GracefulExit' in 'config_file' was changed to '1'.

        Args:
            run_context (RunContext): Context information of the model.
                For more details, please refer to :class:`mindspore.train.RunContext`.
        """
        self._do_step_begin(run_context)

    def on_train_step_end(self, run_context):
        """
        Save checkpoint file or mindir file according to config, and exit the training process.

        Args:
            run_context (RunContext): Include some information of the model.
                For more details, please refer to :class:`mindspore.train.RunContext`.
        """
        self._do_step_end(run_context)

    def on_train_epoch_end(self, run_context):
        """
        When the train epoch end, if received the exit signal,
        set the 'run_context' attribute '_stop_requested' to True.
        Then exit the training process after this epoch training.

        Args:
            run_context (RunContext): Include some information of the model.
                For more details, please refer to :class:`mindspore.train.RunContext`.
        """
        self._do_step_end(run_context)

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
            save_checkpoint(train_net, ckpt_file_name=self.train_name)
        if self.save_mindir:
            inputs = cb_params.train_dataset_element
            export(train_net, *inputs, file_name=self.train_name, file_format='MINDIR')

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
        if os.path.isfile(f"{self.eval_name}.ckpt"):
            load_checkpoint(f"{self.eval_name}.ckpt", net=eval_net)
        elif os.path.isfile(f"{self.train_name}.ckpt"):
            load_checkpoint(f"{self.train_name}.ckpt", net=eval_net)

    def on_eval_step_end(self, run_context):
        """
        When the eval step end, if received the exit signal, set attribute '_stop_requested' of the
        'run_context' to True. Then exit the eval process after this step eval.

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
            save_checkpoint(eval_net, ckpt_file_name=self.eval_name)
        if self.save_mindir:
            inputs = cb_params.eval_dataset_element
            export(eval_net, *inputs, file_name=self.eval_name, file_format='MINDIR')

    def _handle_signal(self, signum, frame):
        """Handle the received signal"""
        logger.debug(f"signum: {signum}, frame: {frame}")
        self.exit = True

    def _do_step_end(self, run_context):
        """
        Save the checkpoint or mindir, and then exit training process.

        Args:
            run_context (RunContext): Include some information of the model.
                For more details, please refer to :class:`mindspore.train.RunContext`.
        """
        with self.lock:
            # save once
            if self.stop_train or not self.need_do_step_end:
                return
            logger.info("Gracefully exiting training process on step end.")
            call_params = run_context.original_args()
            net = call_params.train_network
            for _, param in net.parameters_and_names():
                if param.name == "graceful_exit" and param.asnumpy() == True:  # pylint: disable=C0121
                    logger.warning("Graceful exit is triggered, stop training.")
                    if self.save_ckpt:
                        append_dict = {"epoch_num": call_params.cur_epoch_num,
                                       "step_num": call_params.cur_step_num,
                                       "batch_num": call_params.batch_num}
                        if call_params.loss_scale_mananger is not None:
                            append_dict["loss_scale"] = call_params.loss_scale_mananger.get_loss_scale()
                        if call_params.optimizer is not None:
                            global_step = int(call_params.optimizer.global_step.data)
                        else:
                            global_step = int(call_params.network.optimizer.global_step.data)
                        append_dict["global_step"] = global_step
                        if self.is_distributed:
                            self.integrated_save = _get_parallel_mode() == ParallelMode.AUTO_PARALLEL
                        save_checkpoint(net, self.train_name, integrated_save=self.integrated_save,
                                        append_dict=append_dict)
                    if self.save_mindir:
                        inputs = call_params.train_dataset_element
                        export(net, *inputs, file_name=self.train_name, file_format='MINDIR')
                    run_context.request_stop()
                    self.stop_train = True

    def _do_step_begin(self, run_context):
        """
        Check training process exit configuration at the step begin.

        Args:
            run_context (RunContext): Include some information of the model.
                For more details, please refer to :class:`mindspore.train.RunContext`.
        """
        with self.lock:
            # no env
            if not self.use_graceful:
                return
            if self._check_config_info() or self.exit:
                call_params = run_context.original_args()
                net = call_params.train_network
                for _, param in net.parameters_and_names():
                    if not self.is_distributed and param.name == "graceful_exit":
                        param.set_data(Tensor(True, mstype.bool_))
                        self.need_do_step_end = True
                        break
                    if param.name == "graceful_init":
                        param.set_data(Tensor([1], mstype.int32))
                        self.need_do_step_end = True
                        break

    def _check_config_info(self):
        """check json config info"""
        if self.remote_config_file is not None and os.path.exists(self.remote_config_file):
            with open(self.remote_config_file, "r") as f:
                try:
                    config_info = json.load(f)
                except json.JSONDecodeError as e:
                    logger.warning(f"Parse json file failed: {e}, please check json file: {self.remote_config_file}")
                    return False
                if self.key in config_info and config_info[self.key] == 1:
                    return True
        return False

    def _get_save_path(self, file_name):
        """path to save checkpoint files or mindir files"""
        device_id = context.get_context("device_id")
        if self.save_path is None:
            tmp = os.path.join(os.getcwd(), r"rank_" + str(device_id))
            path_ = _make_directory(tmp)
            return os.path.join(path_, f"{file_name}_train"), os.path.join(path_, f"{file_name}_eval")

        save_path = os.path.join(self.save_path, r"rank_" + str(device_id))
        save_path = _make_directory(save_path)
        return os.path.join(save_path, f"{file_name}_train"), os.path.join(save_path, f"{file_name}_eval")
