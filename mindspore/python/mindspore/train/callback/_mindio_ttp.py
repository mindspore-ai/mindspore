# Copyright 2024 Huawei Technologies Co., Ltd
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
import sys
import traceback
from mindspore.train.serialization import save_checkpoint
from mindspore.parallel._auto_parallel_context import _get_auto_parallel_context
from mindspore.parallel._utils import _get_device_num
from mindspore import _checkparam as Validator
from mindspore.train.callback._callback import Callback
from mindspore import context
import mindspore as ms
from mindspore.communication import get_rank

from mindspore.train._utils import get_parameter_redundancy
from mindspore import log as logger


def _get_dp_from_layout(parameter_layout_dict):
    """ From layout dict get dp and tp """
    pp_num = _get_auto_parallel_context("pipeline_stages")
    dev_num = _get_device_num()
    global_rank = get_rank()
    pipe_size = dev_num // pp_num
    initial_rank = (global_rank // pipe_size) * pipe_size
    parameter_redundancy_dict = get_parameter_redundancy(
        parameter_layout_dict, initial_rank)
    value_len = sys.maxsize
    min_value = ()
    for key, value in parameter_redundancy_dict.items():
        if "accu_grads" in key or "inputs" in key:
            continue
        for item in value:
            if len(item) < value_len and global_rank in item:
                value_len = len(item)
                min_value = item
    return min_value


def _get_ckpt_name(cb_params, is_tmp_file):
    """ common func to generate ckpt name"""
    step_num_in_epoch = int((cb_params.cur_step_num - 1) %
                            cb_params.batch_num + 1)
    tmp = "_tmp" if is_tmp_file else ""
    ckpt_file = f"iteration-{str(cb_params.cur_epoch_num)}_{str(step_num_in_epoch)}{tmp}.ckpt"
    return ckpt_file


def _save_checkpoint_on_failure(ckpt_save_path, run_context):
    """ callback for ttp save ckpt when errors occur """
    cb_params = run_context.original_args()
    cur_ckpt_file = _get_ckpt_name(cb_params, True)
    cur_file = os.path.join(ckpt_save_path, cur_ckpt_file)

    append_dict = {}
    append_dict["epoch_num"] = cb_params.cur_epoch_num
    append_dict["batch_num"] = cb_params.batch_num
    append_dict["step_num"] = cb_params.cur_step_num
    network = cb_params.train_network

    try:
        save_checkpoint(network, cur_file, integrated_save=False,
                        append_dict=append_dict)
    except TypeError:
        logger.critical(" TTP failed to save temp checkpoint file %s. Maybe don't have the permission to write files, "
                        "or the disk is insufficient and so on", cur_ckpt_file)
        return 1
    return 0


def _rename_save_result(ckpt_save_path, run_context):
    """ callback for ttp rename ckpt after ckpt callback finished and successful """
    cb_params = run_context.original_args()
    tmp_name = _get_ckpt_name(cb_params, True)
    fin_name = _get_ckpt_name(cb_params, False)
    try:
        tmp_file = os.path.join(ckpt_save_path, tmp_name)
        fin_file = os.path.join(ckpt_save_path, fin_name)
        os.rename(tmp_file, fin_file)
    except OSError:
        logger.critical(
            f"MindIO adataper rename from {tmp_file} to {fin_file} failed.")
        traceback.print_exc()
        return 1
    return 0


class MindIOTTPAdapter(Callback):
    """
    MindIO TTP Feature Callback

    Args:
        controller_ip (str): ttp controller's ip address
        controller_port (int): ttp controller's ip port
        ckpt_save_path (str): ckpt file save path when failure occurs

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn
        >>> from mindspore.train import Model, MindIOTTPAdapter
        >>> from mindspore import dataset as ds
        >>> ttp_cb = MindIOTTPAdapter("192.168.0.1", 2000, "/tmp/save_checkpoint/")
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
        ...
        >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
        >>> dataset = ds.NumpySlicesDataset(data=data).batch(32)
        >>> model.train(1, dataset, callbacks=[ttp_cb, loss_cb],dataset_sink_mode=False)
    """

    def __init__(self, controller_ip, controller_port, ckpt_save_path):
        super(MindIOTTPAdapter, self).__init__()
        # let it raises errors if not install mindio_ttp package
        from mindio_ttp import framework_ttp as ttp
        self.ttp = ttp
        Validator.check_non_negative_int(controller_port)
        self.has_init = False
        self.enable = False
        mode = context.get_context("mode")
        if context.get_context("device_target") != "Ascend" or mode != context.GRAPH_MODE:
            logger.warning(
                "MindIO adataper only support on Ascend device with GRAPH Mode.")
            return
        if os.getenv("MS_ENABLE_MINDIO_GRACEFUL_EXIT") is None:
            logger.warning("MindIO adataper need custom switch on.")
            return
        self.enable = True
        self._controller_ip = controller_ip
        self._controller_port = controller_port
        self._ckpt_save_path = ckpt_save_path

    def wrapper_ttp_persist(self, func):
        """
        This method used to wrapper ttp exception handler for the input func

        Note:
            This method will check if ttp is enable, if not , will return origin function
        Args:
            ckpt_file_path (str): ckpt file path
            strategy_file_path (str): strategy file path
            net (Cell): network

        """
        if self.enable:
            return self.ttp.ttp_to_persist(func)
        return func

    def _init_ttp(self, run_context):
        """ init mindio ttp """
        dev_num = _get_device_num()

        cb_params = run_context.original_args()
        param_layout_dict = cb_params.train_network.parameter_layout_dict
        dp = _get_dp_from_layout(param_layout_dict)

        self.ttp.ttp_register_save_ckpt_handler(_save_checkpoint_on_failure)
        self.ttp.ttp_register_rename_handler(_rename_save_result)

        world_size = dev_num
        cur_rank = get_rank()
        is_odd = len(dp) % 2
        replica = 2 if is_odd else len(dp) // 2
        enable_local_copy = False
        if cur_rank == 0:
            self.ttp.ttp_init_controller(
                cur_rank, world_size, replica, enable_local_copy)
            self.ttp.ttp_start_controller(
                self._controller_ip, self._controller_port)

        self.ttp.ttp_init_processor(cur_rank, dp, len(
            dp), world_size, replica, enable_local_copy)
        self.ttp.ttp_start_processor(
            self._controller_ip, self._controller_port)

    def on_train_step_end(self, run_context):
        """ override train callback function """
        if self.enable is False:
            return
        pp_num = _get_auto_parallel_context("pipeline_stages")
        if pp_num < 2:
            self.enable = False
            return
        cb_params = run_context.original_args()
        if cb_params.dataset_sink_mode is True and cb_params.sink_size > 1:
            self.enable = False
            return
        if self.has_init is False:
            self.has_init = True
            self._init_ttp(run_context)
        self.ttp.ttp_end_updating_os(cb_params.cur_step_num)
        self.ttp.ttp_set_ckpt_args((self._ckpt_save_path, run_context))

    @staticmethod
    def load_checkpoint_with_backup(ckpt_file_path, strategy_file_path, net):
        """
        Load checkpoint into network, will use strategy file when local ckpt file not found.

        Note:
           This method must be called after communication init function, because we need to know current rank
           and pipeline initial rank.

        Args:
            ckpt_file_path (str): ckpt file path
            strategy_file_path (str): strategy file path
            net (Cell): network

        Examples:
            >>> import numpy as np
            >>> from mindspore import nn
            >>> from mindspore.train import Model, MindIOTTPAdapter
            >>> from mindspore import dataset as ds
            >>> MindIOTTPAdapter.load_checkpoint_with_backup("", "", net)
        """
        try:
            param_dict = ms.load_checkpoint(ckpt_file_path)
        except ValueError:
            dp = _get_dp_from_layout(strategy_file_path)
            rank = get_rank()
            for i in dp:
                if i == rank:
                    continue
                new_ckpt = ckpt_file_path.replace(
                    f"/rank_{rank}/", f"/rank_{str(i)}/")
                try:
                    param_dict = ms.load_checkpoint(new_ckpt)
                except ValueError:
                    param_dict = None
        if param_dict:
            ms.load_param_into_net(net, param_dict)
