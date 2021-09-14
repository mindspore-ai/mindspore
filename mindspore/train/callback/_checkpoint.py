# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Checkpoint related classes and functions."""

import os
import stat
import time

import threading
import mindspore.context as context
from mindspore import log as logger
from mindspore import nn
from mindspore._checkparam import Validator
from mindspore.train._utils import _make_directory
from mindspore.train.serialization import save_checkpoint, _save_graph
from mindspore.parallel._ps_context import _is_role_pserver, _get_ps_mode_rank
from mindspore.parallel._cell_wrapper import destroy_allgather_cell
from ._callback import Callback, set_cur_net
from ...common.tensor import Tensor

_cur_dir = os.getcwd()
_save_dir = _cur_dir
_info_list = ["epoch_num", "step_num"]


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
    The configuration of model checkpoint.

    Note:
        During the training process, if dataset is transmitted through the data channel,
        It is suggested to set 'save_checkpoint_steps' to an integer multiple of loop_size.
        Otherwise, the time to save the checkpoint may be biased.
        It is recommended to set only one save strategy and one keep strategy at the same time.
        If both `save_checkpoint_steps` and `save_checkpoint_seconds` are set,
        `save_checkpoint_seconds` will be invalid.
        If both `keep_checkpoint_max` and `keep_checkpoint_per_n_minutes` are set,
        `keep_checkpoint_per_n_minutes` will be invalid.

    Args:
        save_checkpoint_steps (int): Steps to save checkpoint. Default: 1.
        save_checkpoint_seconds (int): Seconds to save checkpoint.
            Can't be used with save_checkpoint_steps at the same time. Default: 0.
        keep_checkpoint_max (int): Maximum number of checkpoint files can be saved. Default: 5.
        keep_checkpoint_per_n_minutes (int): Save the checkpoint file every `keep_checkpoint_per_n_minutes` minutes.
            Can't be used with keep_checkpoint_max at the same time. Default: 0.
        integrated_save (bool): Whether to merge and save the split Tensor in the automatic parallel scenario.
            Integrated save function is only supported in automatic parallel scene, not supported
            in manual parallel. Default: True.
        async_save (bool): Whether asynchronous execution saves the checkpoint to a file. Default: False.
        saved_network (Cell): Network to be saved in checkpoint file. If the saved_network has no relation
            with the network in training, the initial value of saved_network will be saved. Default: None.
        append_info (list): The information save to checkpoint file. Support "epoch_num", "step_num" and dict.
            The key of dict must be str, the value of dict must be one of int float and bool. Default: None.
        enc_key (Union[None, bytes]): Byte type key used for encryption. If the value is None, the encryption
                                      is not required. Default: None.
        enc_mode (str): This parameter is valid only when enc_key is not set to None. Specifies the encryption
                        mode, currently supports 'AES-GCM' and 'AES-CBC'. Default: 'AES-GCM'.

    Raises:
        ValueError: If input parameter is not the correct type.

    Examples:
        >>> from mindspore import Model, nn
        >>> from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
        >>>
        >>> class LeNet5(nn.Cell):
        ...     def __init__(self, num_class=10, num_channel=1):
        ...         super(LeNet5, self).__init__()
        ...         self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...         self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        ...         self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        ...         self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        ...         self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        ...         self.relu = nn.ReLU()
        ...         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        ...         self.flatten = nn.Flatten()
        ...
        ...     def construct(self, x):
        ...         x = self.max_pool2d(self.relu(self.conv1(x)))
        ...         x = self.max_pool2d(self.relu(self.conv2(x)))
        ...         x = self.flatten(x)
        ...         x = self.relu(self.fc1(x))
        ...         x = self.relu(self.fc2(x))
        ...         x = self.fc3(x)
        ...         return x
        >>>
        >>> net = LeNet5()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
        >>> data_path = './MNIST_Data'
        >>> dataset = create_dataset(data_path)
        >>> config = CheckpointConfig(saved_network=net)
        >>> ckpoint_cb = ModelCheckpoint(prefix='LeNet5', directory='./checkpoint', config=config)
        >>> model.train(10, dataset, callbacks=ckpoint_cb)
    """

    def __init__(self,
                 save_checkpoint_steps=1,
                 save_checkpoint_seconds=0,
                 keep_checkpoint_max=5,
                 keep_checkpoint_per_n_minutes=0,
                 integrated_save=True,
                 async_save=False,
                 saved_network=None,
                 append_info=None,
                 enc_key=None,
                 enc_mode='AES-GCM'):

        if save_checkpoint_steps is not None:
            save_checkpoint_steps = Validator.check_non_negative_int(save_checkpoint_steps)
        if save_checkpoint_seconds is not None:
            save_checkpoint_seconds = Validator.check_non_negative_int(save_checkpoint_seconds)
        if keep_checkpoint_max is not None:
            keep_checkpoint_max = Validator.check_non_negative_int(keep_checkpoint_max)
        if keep_checkpoint_per_n_minutes is not None:
            keep_checkpoint_per_n_minutes = Validator.check_non_negative_int(keep_checkpoint_per_n_minutes)

        if saved_network is not None and not isinstance(saved_network, nn.Cell):
            raise TypeError(f"The type of saved_network must be None or Cell, but got {str(type(saved_network))}.")

        if not save_checkpoint_steps and not save_checkpoint_seconds and \
                not keep_checkpoint_max and not keep_checkpoint_per_n_minutes:
            raise ValueError("The input arguments 'save_checkpoint_steps', 'save_checkpoint_seconds', "
                             "'keep_checkpoint_max' and 'keep_checkpoint_per_n_minutes' can't be all None or 0.")

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

        self._integrated_save = Validator.check_bool(integrated_save)
        self._async_save = Validator.check_bool(async_save)
        self._saved_network = saved_network
        self._append_dict = self._handle_append_info(append_info)
        self._enc_key = Validator.check_isinstance('enc_key', enc_key, (type(None), bytes))
        self._enc_mode = Validator.check_isinstance('enc_mode', enc_mode, str)

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

    @property
    def integrated_save(self):
        """Get the value of _integrated_save."""
        return self._integrated_save

    @property
    def async_save(self):
        """Get the value of _async_save."""
        return self._async_save

    @property
    def saved_network(self):
        """Get the value of _saved_network"""
        return self._saved_network

    @property
    def enc_key(self):
        """Get the value of _enc_key"""
        return self._enc_key

    @property
    def enc_mode(self):
        """Get the value of _enc_mode"""
        return self._enc_mode

    @property
    def append_dict(self):
        """Get the value of append_dict."""
        return self._append_dict

    def get_checkpoint_policy(self):
        """Get the policy of checkpoint."""
        checkpoint_policy = {'save_checkpoint_steps': self.save_checkpoint_steps,
                             'save_checkpoint_seconds': self.save_checkpoint_seconds,
                             'keep_checkpoint_max': self.keep_checkpoint_max,
                             'keep_checkpoint_per_n_minutes': self.keep_checkpoint_per_n_minutes,
                             'saved_network': self.saved_network}

        return checkpoint_policy

    @staticmethod
    def _handle_append_info(append_info):
        """Handle ckpt append info."""
        if append_info is None or append_info == []:
            return None
        if not isinstance(append_info, list):
            raise TypeError(f"The type of 'append_info' must list, but got {str(type(append_info))}.")
        handle_append_info = {}
        if "epoch_num" in append_info:
            handle_append_info["epoch_num"] = 0
        if "step_num" in append_info:
            handle_append_info["step_num"] = 0
        dict_num = 0
        for element in append_info:
            if not isinstance(element, str) and not isinstance(element, dict):
                raise TypeError(f"The type of append_info element must be str or dict, but got {str(type(element))}.")
            if isinstance(element, str) and element not in _info_list:
                raise TypeError(f"The type of append_info element must be in {_info_list}, but got {element}.")
            if isinstance(element, dict):
                dict_num += 1
                if dict_num > 1:
                    raise TypeError(f"The element of append_info must has only one dict.")
                for key, value in element.items():
                    if isinstance(key, str) and isinstance(value, (int, float, bool)):
                        handle_append_info[key] = value
                    else:
                        raise TypeError(f"The type of dict in append_info must be key: str, value: int or float.")

        return handle_append_info


class ModelCheckpoint(Callback):
    """
    The checkpoint callback class.

    It is called to combine with train process and save the model and network parameters after training.

    Note:
        In the distributed training scenario, please specify different directories for each training process
        to save the checkpoint file. Otherwise, the training may fail.

    Args:
        prefix (str): The prefix name of checkpoint files. Default: "CKP".
        directory (str): The path of the folder which will be saved in the checkpoint file.
            By default, the file is saved in the current directory. Default: None.
        config (CheckpointConfig): Checkpoint strategy configuration. Default: None.

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

        if not isinstance(prefix, str) or prefix.find('/') >= 0:
            raise ValueError("'Prefix' {} for checkpoint file name is invalid, 'prefix' must be "
                             "str and does not contain '/', please check and correct it and then "
                             "continue".format(prefix))
        self._prefix = prefix

        if directory is not None:
            self._directory = _make_directory(directory)
        else:
            self._directory = _cur_dir

        if config is None:
            self._config = CheckpointConfig()
        else:
            if not isinstance(config, CheckpointConfig):
                raise TypeError("The argument 'config' should be CheckpointConfig type.")
            self._config = config

        # get existing checkpoint files
        self._manager = CheckpointManager()
        self._prefix = _chg_ckpt_file_name_if_same_exist(self._directory, self._prefix)
        self._append_dict = self._config.append_dict or {}
        self._append_epoch_num = self._append_dict["epoch_num"] if "epoch_num" in self._append_dict else 0
        self._append_step_num = self._append_dict["step_num"] if "step_num" in self._append_dict else 0
        self._graph_saved = False
        self._need_flush_from_cache = True

    def step_end(self, run_context):
        """
        Save the checkpoint at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        if _is_role_pserver():
            self._prefix = "PServer_" + str(_get_ps_mode_rank()) + "_" + self._prefix
        cb_params = run_context.original_args()
        _make_directory(self._directory)
        # save graph (only once)
        if not self._graph_saved:
            graph_file_name = os.path.join(self._directory, self._prefix + '-graph.meta')
            if os.path.isfile(graph_file_name) and context.get_context("mode") == context.GRAPH_MODE:
                os.remove(graph_file_name)
            _save_graph(cb_params.train_network, graph_file_name)
            self._graph_saved = True
        thread_list = threading.enumerate()
        for thread in thread_list:
            if thread.getName() == "asyn_save_ckpt":
                thread.join()
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

        thread_list = threading.enumerate()
        for thread in thread_list:
            if thread.getName() == "asyn_save_ckpt":
                thread.join()

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

        return False

    def _save_ckpt(self, cb_params, force_to_save=False):
        """Save checkpoint files."""
        if cb_params.cur_step_num == self._last_triggered_step:
            return

        # if param is cache enable, flush data from cache to host before save_ckpt
        if self._need_flush_from_cache:
            self._flush_from_cache(cb_params)

        save_ckpt = self._check_save_ckpt(cb_params, force_to_save)
        step_num_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)

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
            self._last_time_for_keep = time.time()
            self._last_triggered_step = cb_params.cur_step_num

            if context.get_context("enable_ge"):
                set_cur_net(cb_params.train_network)
                cb_params.train_network.exec_checkpoint_graph()
            if "epoch_num" in self._append_dict:
                self._append_dict["epoch_num"] = self._append_epoch_num + cb_params.cur_epoch_num
            if "step_num" in self._append_dict:
                self._append_dict["step_num"] = self._append_step_num + cb_params.cur_step_num
            network = self._config.saved_network if self._config.saved_network is not None else cb_params.train_network
            save_checkpoint(network, cur_file, self._config.integrated_save, self._config.async_save,
                            self._append_dict, self._config.enc_key, self._config.enc_mode)

            self._latest_ckpt_file_name = cur_file

    def _flush_from_cache(self, cb_params):
        """Flush cache data to host if tensor is cache enable."""
        has_cache_params = False
        params = cb_params.train_network.get_parameters()
        for param in params:
            if param.cache_enable:
                has_cache_params = True
                Tensor(param).flush_from_cache()
        if not has_cache_params:
            self._need_flush_from_cache = False

    @property
    def latest_ckpt_file_name(self):
        """Return the latest checkpoint path and file name."""
        return self._latest_ckpt_file_name


class CheckpointManager:
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
            if os.path.splitext(filename)[-1] == ".ckpt" and filename.startswith(prefix + "-"):
                mid_name = filename[len(prefix):-5]
                flag = not (True in [char.isalpha() for char in mid_name])
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
        del_list = []
        oldest_file = ''
        oldest_time = cur_time
        for ck_file in self._ckpoint_filelist:
            modify_time = os.path.getmtime(ck_file)
            if cur_time - modify_time < 60 * minutes:
                del_list.append(ck_file)

                if modify_time < oldest_time:
                    oldest_time = modify_time
                    oldest_file = ck_file

        for mv_file in del_list:
            if mv_file == oldest_file:
                continue
            self.remove_ckpoint_file(mv_file)
