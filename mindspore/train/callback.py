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

import os
import stat
import shutil
import time
import numpy as np

import mindspore.context as context
from mindspore.train.serialization import _exec_save_checkpoint, _fill_param_into_net, _save_graph
from mindspore.train._utils import _make_directory
from mindspore import log as logger
from mindspore._checkparam import check_int_non_negative
from mindspore.common.tensor import Tensor
from .summary.summary_record import _cache_summary_tensor_data


__all__ = ["Callback", "LossMonitor", "TimeMonitor", "ModelCheckpoint", "SummaryStep", "CheckpointConfig", "RunContext"]


_cur_dir = os.getcwd()
_cur_net = None
_save_dir = _cur_dir


class _CheckpointManager:
    """Manage checkpoint files according to train_config of checkpoint."""
    def __init__(self):
        self._ckpoint_filelist = []

    @property
    def ckpoint_filelist(self):
        """Get all the related checkpoint files managed here."""
        return self._ckpoint_filelist

    @property
    def ckpoint_num(self):
        """Get the number of the related checkpoint files managed here."""
        return len(self._ckpoint_filelist)

    def update_ckpoint_filelist(self, directory, prefix):
        """Update the checkpoint file list."""
        self._ckpoint_filelist = []
        files = os.listdir(directory)
        for filename in files:
            if os.path.splitext(filename)[-1] == ".ckpt" and filename.startswith(prefix):
                mid_name = filename[len(prefix):-5]
                flag = True
                for char in mid_name:
                    if char.isalpha():
                        flag = False
                if flag:
                    self._ckpoint_filelist.append(directory + '/' + filename)

    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
            self._ckpoint_filelist.remove(file_name)
        except OSError:
            logger.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            logger.warning("ValueError, failed to remove the older ckpt file %s.", file_name)

    def remove_oldest_ckpoint_file(self):
        """Remove the oldest checkpoint file from this checkpoint manager and also from the directory."""
        ckpoint_files = sorted(self._ckpoint_filelist, key=os.path.getmtime)
        self.remove_ckpoint_file(ckpoint_files[0])

    def keep_one_ckpoint_per_minutes(self, minutes, cur_time):
        """Only keep the latest one ckpt file per minutes, remove other files generated in [last_time, cur_time]."""
        movs = []
        oldest_file = ''
        oldest_time = cur_time
        for ck_file in self._ckpoint_filelist:
            modify_time = os.path.getmtime(ck_file)
            if cur_time - modify_time < 60 * minutes:
                movs.append(ck_file)

                if modify_time < oldest_time:
                    oldest_time = modify_time
                    oldest_file = ck_file

        for mv_file in movs:
            if mv_file == oldest_file:
                continue
            self.remove_ckpoint_file(mv_file)


def _check_file_name_prefix(file_name_prefix):
    """
    Check file name valid or not.

    File name can't include '/'. This file name naming convention only apply to Linux.
    """
    if not isinstance(file_name_prefix, str) or file_name_prefix.find('/') >= 0:
        return False
    return True


def _chg_ckpt_file_name_if_same_exist(directory, prefix):
    """Check if there is a file with the same name."""
    files = os.listdir(directory)
    suffix_num = 0
    pre_len = len(prefix)
    for filename in files:
        name_ext = os.path.splitext(filename)
        if name_ext[-1] != ".ckpt":
            continue
        # find same prefix file
        if filename.find(prefix) == 0 and not filename[pre_len].isalpha():
            # add the max suffix + 1
            index = filename[pre_len:].find("-")
            if index == 0:
                suffix_num = max(suffix_num, 1)
            elif index != -1:
                num = filename[pre_len+1:pre_len+index]
                if num.isdigit():
                    suffix_num = max(suffix_num, int(num)+1)

    if suffix_num != 0:
        prefix = prefix + "_" + str(suffix_num)

    return prefix


class CheckpointConfig:
    """
    The config for model checkpoint.

    Args:
        save_checkpoint_steps (int): Steps to save checkpoint. Default: 1.
        save_checkpoint_seconds (int): Seconds to save checkpoint. Default: 0.
            Can't be used with save_checkpoint_steps at the same time.
        keep_checkpoint_max (int): Maximum step to save checkpoint. Default: 5.
        keep_checkpoint_per_n_minutes (int): Keep one checkpoint every n minutes. Default: 0.
            Can't be used with keep_checkpoint_max at the same time.

    Raises:
        ValueError: If the input_param is None or 0.

    Examples:
        >>> config = CheckpointConfig()
        >>> ckpoint_cb = ModelCheckpoint(prefix="ck_prefix", directory='./', config=config)
        >>> model.train(10, dataset, callbacks=ckpoint_cb)
    """
    def __init__(self,
                 save_checkpoint_steps=1,
                 save_checkpoint_seconds=0,
                 keep_checkpoint_max=5,
                 keep_checkpoint_per_n_minutes=0):

        if not save_checkpoint_steps and not save_checkpoint_seconds and \
                not keep_checkpoint_max and not keep_checkpoint_per_n_minutes:
            raise ValueError("The input_param can't be all None or 0")

        if save_checkpoint_steps:
            save_checkpoint_steps = check_int_non_negative(save_checkpoint_steps)
        if save_checkpoint_seconds:
            save_checkpoint_seconds = check_int_non_negative(save_checkpoint_seconds)
        if keep_checkpoint_max:
            keep_checkpoint_max = check_int_non_negative(keep_checkpoint_max)
        if keep_checkpoint_per_n_minutes:
            keep_checkpoint_per_n_minutes = check_int_non_negative(keep_checkpoint_per_n_minutes)

        self._save_checkpoint_steps = save_checkpoint_steps
        self._save_checkpoint_seconds = save_checkpoint_seconds
        if self._save_checkpoint_steps and self._save_checkpoint_steps > 0:
            self._save_checkpoint_seconds = None

        self._keep_checkpoint_max = keep_checkpoint_max
        self._keep_checkpoint_per_n_minutes = keep_checkpoint_per_n_minutes
        if self._keep_checkpoint_max and self._keep_checkpoint_max > 0:
            self._keep_checkpoint_per_n_minutes = None
        else:
            if not self._keep_checkpoint_per_n_minutes or self._keep_checkpoint_per_n_minutes == 0:
                self._keep_checkpoint_max = 1

    @property
    def save_checkpoint_steps(self):
        """Get the value of _save_checkpoint_steps."""
        return self._save_checkpoint_steps

    @property
    def save_checkpoint_seconds(self):
        """Get the value of _save_checkpoint_seconds."""
        return self._save_checkpoint_seconds

    @property
    def keep_checkpoint_max(self):
        """Get the value of _keep_checkpoint_max."""
        return self._keep_checkpoint_max

    @property
    def keep_checkpoint_per_n_minutes(self):
        """Get the value of _keep_checkpoint_per_n_minutes."""
        return self._keep_checkpoint_per_n_minutes

    def get_checkpoint_policy(self):
        """Get the policy of checkpoint."""
        checkpoint_policy = {'save_checkpoint_steps': self._save_checkpoint_steps,
                             'save_checkpoint_seconds': self._save_checkpoint_seconds,
                             'keep_checkpoint_max': self._keep_checkpoint_max,
                             'keep_checkpoint_per_n_minutes': self._keep_checkpoint_per_n_minutes}

        return checkpoint_policy


def _set_cur_net(net):
    """
    Set current net for which we are using to save checkpoint.

    Args:
        net (Cell): train network
    """
    global _cur_net
    _cur_net = net


def _checkpoint_cb_for_save_op(parameter_list):
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
    _set_cur_net(None)
    return True


def _summary_cb_for_save_op(summary_list):
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


def _build_callbacks(callbacks):
    """
    Contain a list of callback.

    Args:
        callbacks (list): Callback functions list, Support None, a single Callback object, or a list.

    Returns:
        List, a list of callback functions.
    """
    if callbacks:
        if isinstance(callbacks, tuple):
            raise TypeError("Callbacks cannot be a tuple. Please check it.")
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
    else:
        callbacks = []

    excute_callbacks = []
    for cb in callbacks:
        if cb is None or not isinstance(cb, Callback):
            raise TypeError("Callback must inheriting base class Callback. Some callback is Wrong. Please check it.")
        excute_callbacks.append(cb)

    return _ListCallback(excute_callbacks)


class _ListCallback:
    """
    Sequential execution of callback functions.

    Execute Callback functions at certain points.

    Args:
        callbacks (list): Callback functions list.
    """
    def __init__(self, callbacks):
        super(_ListCallback, self).__init__()
        self._callbacks = callbacks

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


class Callback:
    """
    Abstract base class used to build a callback function.

    Callback function will execution some operating to the current step or epoch.

    Examples:
        >>> class Print_info(Callback):
        >>>     def step_end(self, run_context):
        >>>         cb_params = run_context.original_args()
        >>>         print(cb_params.cur_epoch_num)
        >>>         print(cb_params.cur_step_num)
        >>>
        >>> print_cb = Print_info()
        >>> model.train(epoch, dataset, callback=print_cb)
    """
    def __init__(self):
        pass

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


class SummaryStep(Callback):
    """
    The summary callback class.

    Args:
        summary (Object): Summary recode object.
        flush_step (int): Number of interval steps to execute. Default: 10.
    """
    def __init__(self, summary, flush_step=10):
        super(SummaryStep, self).__init__()
        if not isinstance(flush_step, int) or isinstance(flush_step, bool) or flush_step <= 0:
            raise ValueError("`flush_step` should be int and greater than 0")
        self._summary = summary
        self._flush_step = flush_step

    def step_end(self, run_context):
        """
        Save summary.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_step_num % self._flush_step == 0:
            self._summary.record(cb_params.cur_step_num, cb_params.train_network)

    @property
    def summary_file_name(self):
        return self._summary.full_file_name


class _InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class RunContext:
    """
    Provides information about the model.

    Run call being made. Provides information about original request to model function.
    callback objects can stop the loop by calling request_stop() of run_context.

    Args:
        original_args (dict): Holding the related information of model etc.
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
           _InternalCallbackParam, a object holding the original arguments of model.
        """
        return self._original_args

    def request_stop(self):
        """
        Sets stop requested during training.

        Callbacks can use this function to request stop of iterations.
        model.train() checks whether this is called or not.
        """
        self._stop_requested = True

    def get_stop_requested(self):
        """
        Returns whether a stop is requested or not.

        Returns:
            bool, if true, model.train() stops iterations.
        """
        return self._stop_requested


class ModelCheckpoint(Callback):
    """
    The checkpoint callback class.

    It is called to combine with train process and save the model and network parameters after traning.

    Args:
        prefix (str): Checkpoint files names prefix. Default: "CKP".
        directory (str): Lolder path into which checkpoint files will be saved. Default: None.
        config (CheckpointConfig): Checkpoint strategy config. Default: None.

    Raises:
        ValueError: If the prefix is invalid.
        TypeError: If the config is not CheckpointConfig type.
    """
    def __init__(self, prefix='CKP', directory=None, config=None):
        super(ModelCheckpoint, self).__init__()
        self._latest_ckpt_file_name = ""
        self._init_time = time.time()
        self._last_time = time.time()
        self._last_time_for_keep = time.time()
        self._last_triggered_step = 0

        if _check_file_name_prefix(prefix):
            self._prefix = prefix
        else:
            raise ValueError("Prefix {} for checkpoint file name invalid, "
                             "please check and correct it and then continue.".format(prefix))

        if directory:
            self._directory = _make_directory(directory)
        else:
            self._directory = _cur_dir

        if config is None:
            self._config = CheckpointConfig()
        else:
            if not isinstance(config, CheckpointConfig):
                raise TypeError("config should be CheckpointConfig type.")
            self._config = config

        # get existing checkpoint files
        self._manager = _CheckpointManager()
        self._prefix = _chg_ckpt_file_name_if_same_exist(self._directory, self._prefix)
        self._graph_saved = False

    def step_end(self, run_context):
        """
        Save the checkpoint at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        # save graph (only once)
        if not self._graph_saved:
            graph_file_name = os.path.join(self._directory, self._prefix + '-graph.meta')
            _save_graph(cb_params.train_network, graph_file_name)
            self._graph_saved = True
        self._save_ckpt(cb_params)

    def end(self, run_context):
        """
        Save the last checkpoint after training finished.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        _to_save_last_ckpt = True
        self._save_ckpt(cb_params, _to_save_last_ckpt)

        from mindspore.parallel._cell_wrapper import destroy_allgather_cell
        destroy_allgather_cell()

    def _check_save_ckpt(self, cb_params, force_to_save):
        """Check whether save checkpoint files or not."""
        if self._config.save_checkpoint_steps and self._config.save_checkpoint_steps > 0:
            if cb_params.cur_step_num >= self._last_triggered_step + self._config.save_checkpoint_steps \
                 or force_to_save is True:
                return True
        elif self._config.save_checkpoint_seconds and self._config.save_checkpoint_seconds > 0:
            self._cur_time = time.time()
            if (self._cur_time - self._last_time) > self._config.save_checkpoint_seconds or force_to_save is True:
                self._last_time = self._cur_time
                return True
        else:
            if cb_params.cur_step_num == cb_params.step_num:
                return True
        return False

    def _save_ckpt(self, cb_params, force_to_save=False):
        """Save checkpoint files."""
        if cb_params.cur_step_num == self._last_triggered_step:
            return

        save_ckpt = self._check_save_ckpt(cb_params, force_to_save)
        step_num_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if save_ckpt:
            cur_ckpoint_file = self._prefix + "-" + str(cb_params.cur_epoch_num) + "_" \
                               + str(step_num_in_epoch) + ".ckpt"
            # update checkpoint file list.
            self._manager.update_ckpoint_filelist(self._directory, self._prefix)
            # keep checkpoint files number equal max number.
            if self._config.keep_checkpoint_max and 0 < self._config.keep_checkpoint_max <= self._manager.ckpoint_num:
                self._manager.remove_oldest_ckpoint_file()
            elif self._config.keep_checkpoint_per_n_minutes and self._config.keep_checkpoint_per_n_minutes > 0:
                self._cur_time_for_keep = time.time()
                if (self._cur_time_for_keep - self._last_time_for_keep) \
                        < self._config.keep_checkpoint_per_n_minutes * 60:
                    self._manager.keep_one_ckpoint_per_minutes(self._config.keep_checkpoint_per_n_minutes,
                                                               self._cur_time_for_keep)

            # generate the new checkpoint file and rename it.
            global _save_dir
            _save_dir = self._directory
            cur_file = os.path.join(self._directory, cur_ckpoint_file)
            tmp_ckpt_file_name_for_cur_process = str(os.getpid()) + "-" + 'parameters.ckpt'
            gen_file = os.path.join(_save_dir, tmp_ckpt_file_name_for_cur_process)
            self._last_time_for_keep = time.time()
            self._last_triggered_step = cb_params.cur_step_num

            if context.get_context("enable_ge"):
                _set_cur_net(cb_params.train_network)
                cb_params.train_network.exec_checkpoint_graph()

            _exec_save_checkpoint(cb_params.train_network, gen_file)

            if os.path.exists(gen_file):
                shutil.move(gen_file, cur_file)
            self._latest_ckpt_file_name = cur_file

    @property
    def latest_ckpt_file_name(self):
        """Return the latest checkpoint path and file name."""
        return self._latest_ckpt_file_name


class LossMonitor(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.

    Raises:
        ValueError: If print_step is not int or less than zero.
    """
    def __init__(self, per_print_times=1):
        super(LossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training."
                             .format(cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss), flush=True)


class TimeMonitor(Callback):
    def __init__(self, data_size):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()
   
    def epoch_end(self, run_context):
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / self.data_size
        print("epoch time: {0}, per step time: {1}".format(epoch_mseconds, per_step_mseconds), flush=True)

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        step_mseconds = (time.time() - self.step_time) * 1000
        print('step time', step_mseconds, flush=True)

