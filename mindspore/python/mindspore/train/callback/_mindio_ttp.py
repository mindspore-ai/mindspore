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
import copy
from mindspore.train.serialization import save_checkpoint, _convert_cell_param_and_names_to_dict, _get_merged_param_data
from mindspore.parallel._auto_parallel_context import _get_auto_parallel_context
from mindspore.parallel._utils import _get_device_num
from mindspore import _checkparam as Validator
from mindspore.train.callback._callback import Callback
from mindspore.common.tensor import Tensor
from mindspore import context
import mindspore as ms
from mindspore.communication import get_rank
from mindspore.parallel.checkpoint_transform import sync_pipeline_shared_parameters

from mindspore.train._utils import get_parameter_redundancy
from mindspore import log as logger
from mindspore.parallel._utils import _is_in_auto_parallel_mode
from mindspore.common.api import _get_parameter_layout


def _get_dp_from_layout(parameter_layout_dict):
    """ Get dp and tp from layout dict. """
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


def _get_ckpt_dir(append_dict, ckpt_save_path, is_tmp_file):
    """ Common func to generate ckpt dir name."""
    tmp = "_tmp" if is_tmp_file else ""
    mid_dir = f"ttp_saved_checkpoints-{str(append_dict['cur_epoch_num'])}_{str(append_dict['cur_step_num'])}{tmp}"
    return os.path.join(ckpt_save_path, mid_dir)


def _flush_from_cache(cb_params):
    """ Flush cache data to host if tensor is cache enable."""
    params = cb_params.train_network.get_parameters()
    for param in params:
        if param.cache_enable:
            Tensor(param).flush_from_cache()


def _save_checkpoint_on_failure(save_rank, step, rank_list, save_args):
    """ Callback used for TTP save ckpt function when errors occur."""
    logger.info("Enter _save_checkpoint_on_failure function")
    ckpt_save_path, save_params, append_dict = save_args
    ckpt_file = f"iteration-{str(append_dict['cur_epoch_num'])}_{str(append_dict['cur_step_num'])}.ckpt"
    cur_ckpt_dir = _get_ckpt_dir(
        append_dict, ckpt_save_path, True) + "/rank_" + str(save_rank)
    os.makedirs(cur_ckpt_dir)
    cur_file = os.path.join(cur_ckpt_dir, ckpt_file)
    save_checkpoint(save_params, cur_file,
                    integrated_save=False, append_dict=append_dict)
    logger.info("Finish _save_checkpoint_on_failure function")


def _convert_net_to_param_list(save_obj):
    """Convert nn.Cell to param_list."""
    sync_pipeline_shared_parameters(save_obj)
    param_list = []
    parameter_layout_dict = save_obj.parameter_layout_dict
    if _is_in_auto_parallel_mode() and not parameter_layout_dict:
        parameter_layout_dict = _get_parameter_layout()
    if not _is_in_auto_parallel_mode():
        save_obj.init_parameters_data()
    param_dict = _convert_cell_param_and_names_to_dict(save_obj, None)
    for (key, value) in param_dict.items():
        each_param = {"name": key}
        param_data = Tensor(value.asnumpy())
        # in automatic model parallel scenario, some parameters were split to all the devices,
        # which should be combined before saving
        if key in parameter_layout_dict:
            param_data = _get_merged_param_data(
                save_obj, parameter_layout_dict, key, param_data, False)
        each_param["data"] = param_data
        param_list.append(each_param)
    return param_list


def _rename_save_result(rename_args):
    """ Callback used for TTP rename function after ckpt save callback was finished and successful."""
    logger.info("Enter _rename_save_result function")
    ckpt_save_path, _, append_dict = rename_args

    tmp_dir = _get_ckpt_dir(append_dict, ckpt_save_path, True)
    fin_dir = _get_ckpt_dir(append_dict, ckpt_save_path, False)

    os.rename(tmp_dir, fin_dir)
    logger.info("Finish _rename_save_result function")


class MindIOTTPAdapter(Callback):
    """
    This callback is used to enable the feature
    `MindIO TTP <https://www.hiascend.com/document/detail/zh/mindx-dl/60rc1/mindio/mindiottp/mindiottp001.html>`_.
    This callback will execute TTP operations during training process, such as TTP init, report and exception handle.

    Note:
        Required for Ascend GE LazyInline mode only. And pipline size must be greater than 1.

    Args:
        controller_ip (str): TTP controller's ip address, used for init TTP controller.
        controller_port (int): TTP controller's ip port, used for init TTP controller and processor.
        ckpt_save_path (str): Checkpoint save directory when failure occurs, checkpoint file will save to directory
           named ttp_saved_checkpoints-{cur_epoch_num}_{cur_step_num} under this directory.

    Raises:
        Exception: TTP init failed.
        ModuleNotFoundError: Mindio TTP whl package is not installed.

    Examples:
        >>> import numpy as np
        >>> import os
        >>> import math
        >>> import mindspore as ms
        >>> import mindspore.dataset as ds
        >>> from mindspore import nn, ops, Parameter, train
        >>> from mindspore.communication import init
        >>> from mindspore.common.initializer import initializer, HeUniform
        >>> from mindspore.train import Model, MindIOTTPAdapter
        >>> from mindspore import dataset as ds
        >>> ms.set_context(mode=ms.GRAPH_MODE, jit_level='O2')
        >>> ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=2)
        >>> init()
        >>> ms.set_seed(1)
        >>> ms.set_auto_parallel_context(strategy_ckpt_config={"save_file":
        >>>                             "./src_pipeline_strategys/src_strategy_{}.ckpt".format(get_rank())})
        >>> class MatMulCell(nn.Cell):
        ...     def __init__(self, param=None, shape=None):
        ...         super().__init__()
        ...         if shape is None:
        ...             shape = [28 * 28, 512]
        ...         weight_init = HeUniform(math.sqrt(5))
        ...         self.param = Parameter(initializer(weight_init, shape), name="param")
        ...         if param is not None:
        ...             self.param = param
        ...         self.print = ops.Print()
        ...         self.matmul = ops.MatMul()
        ...
        ...     def construct(self, x):
        ...         out = self.matmul(x, self.param)
        ...         self.print("out is:", out)
        ...         return out
        >>>
        >>> class Network(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.flatten = nn.Flatten()
        ...         self.layer1 = MatMulCell()
        ...         self.relu1 = nn.ReLU()
        ...         self.layer2 = nn.Dense(512, 512)
        ...         self.relu2 = nn.ReLU()
        ...         self.layer3 = nn.Dense(512, 10)
        ...
        ...     def construct(self, x):
        ...         x = self.flatten(x)
        ...         x = self.layer1(x)
        ...         x = self.relu1(x)
        ...         x = self.layer2(x)
        ...         x = self.relu2(x)
        ...         logits = self.layer3(x)
        ...         return logits
        >>>
        >>> net = Network()
        >>> net.layer1.pipeline_stage = 0
        >>> net.relu1.pipeline_stage = 0
        >>> net.layer2.pipeline_stage = 0
        >>> net.relu2.pipeline_stage = 1
        >>> net.layer3.pipeline_stage = 1
        >>>
        >>> def create_dataset(batch_size):
        ...     dataset_path = os.getenv("DATA_PATH")
        ...     dataset = ds.MnistDataset(dataset_path)
        ...     image_transforms = [
        ...         ds.vision.Rescale(1.0 / 255.0, 0),
        ...         ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ...         ds.vision.HWC2CHW()
        ...     ]
        ...     label_transform = ds.transforms.TypeCast(ms.int32)
        ...     dataset = dataset.map(image_transforms, 'image')
        ...     dataset = dataset.map(label_transform, 'label')
        ...     dataset = dataset.batch(batch_size)
        ...     return dataset
        >>>
        >>> data_set = create_dataset(32)
        >>>
        >>> optimizer = nn.SGD(net.trainable_params(), 1e-2)
        >>> loss_fn = nn.CrossEntropyLoss()
        >>>
        >>> net_with_loss = nn.PipelineCell(nn.WithLossCell(net, loss_fn), 4)
        >>> net_with_loss.set_train()
        >>> model = Model(net_with_loss, optimizer=optimizer)
        >>> ttp_cb = MindIOTTPAdapter("192.168.0.1", 2000, "./ttp_checkpoint/")
        >>> loss_cb = train.LossMonitor(1)
        >>> model.train(1, dataset, callbacks=[ttp_cb, loss_cb])
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
        if os.getenv("MS_ENABLE_MINDIO_GRACEFUL_EXIT") != "true":
            logger.warning("MindIO adataper need custom switch on.")
            return
        ttp_lib_path = os.getenv("MS_MINDIO_TTP_LIB_PATH")
        if ttp_lib_path is None or os.path.isfile(ttp_lib_path) is False:
            logger.warning(
                "MindIO adataper switch on, but ttp library path is not correct.")
            return
        self.enable = True
        self._controller_ip = controller_ip
        self._controller_port = controller_port
        self._ckpt_save_path = ckpt_save_path

    def wrapper_ttp_persist(self, func):
        """
        This method is used to wrapper TTP exception handler for the input func.

        Args:
            func (function): train method that need to be wrapper.

        Returns:
            Function, if the TTP is enabled, return the encapsulated function,
            otherwise the original function is returned.

        """
        if self.enable:
            return self.ttp.ttp_to_persist(func)
        return func

    def _init_ttp(self, run_context):
        """ Init Mindio TTP, used internal. """
        logger.info("Begin to init ttp.")
        dev_num = _get_device_num()

        cb_params = run_context.original_args()
        param_layout_dict = cb_params.train_network.parameter_layout_dict
        dp = _get_dp_from_layout(param_layout_dict)
        logger.info("Init TTP with dp: {}.".format(dp))

        self.ttp.ttp_register_save_ckpt_handler(_save_checkpoint_on_failure)
        self.ttp.ttp_register_rename_handler(_rename_save_result)

        world_size = dev_num
        cur_rank = get_rank()
        is_odd = len(dp) % 2
        replica = 2 if is_odd else len(dp) // 2
        enable_local_copy = False
        if cur_rank == 0:
            logger.info("Begin to start ttp controller.")
            self.ttp.ttp_init_controller(
                cur_rank, world_size, replica, enable_local_copy)
            self.ttp.ttp_start_controller(
                self._controller_ip, self._controller_port)
            logger.info("Finish start ttp controller.")

        logger.info("Begin to start ttp processor.")
        self.ttp.ttp_init_processor(cur_rank, dp, len(
            dp), world_size, replica, enable_local_copy)
        self.ttp.ttp_start_processor(
            self._controller_ip, self._controller_port)
        logger.info("Finished start ttp processor.")

        logger.info("Finish init ttp.")

    def on_train_step_end(self, run_context):
        """
        Init TTP Controller only once after first step finished.
        And report status to MindIO TTP after every step finished.

        Args:
            run_context (RunContext): Context of the train running. Refer to
                                      :class:`mindspore.train.RunContext` for detail.

        """

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
        _flush_from_cache(cb_params)
        cur_rank = get_rank()
        append_dict = {}
        append_dict["cur_epoch_num"] = cb_params.cur_epoch_num
        append_dict["cur_step_num"] = int(
            (cb_params.cur_step_num - 1) % cb_params.batch_num + 1)
        append_dict["cur_rank"] = cur_rank
        append_dict["batch_num"] = cb_params.batch_num
        append_dict["global_step"] = cb_params.cur_step_num

        save_params = _convert_net_to_param_list(cb_params.train_network)
        save_params_copy = copy.deepcopy(save_params)

        logger.info("Set ckpt args to TTP.")
        self.ttp.ttp_set_ckpt_args(
            (self._ckpt_save_path, save_params_copy, append_dict))
        logger.info("Set optimizer finish step status to TTP.")
        self.ttp.ttp_end_updating_os(cb_params.cur_step_num)

    @staticmethod
    def load_checkpoint_with_backup(ckpt_file_path, strategy_file_path, net):
        """
        Load checkpoint into network, and use strategy file to find backup checkpoint file
        when origin checkpoint file not found.

        Note:
           This API must be called after the communication is initialized because the cluster information
           needs to be obtained internally.

        Args:
            ckpt_file_path (str): the checkpoint file to be loaded.
            strategy_file_path (str): strategy file path for current rank.
            net (Cell): network that needs to load checkpoint.

        Returns:
            Dict, checkpoint weights after loaded.

        Raises:
            ValueError: Failed to load the checkpoint file.

        Examples:
            >>> import numpy as np
            >>> from mindspore import nn
            >>> from mindspore.train import Model, MindIOTTPAdapter
            >>> from mindspore import dataset as ds
            >>> ms.set_context(mode=ms.GRAPH_MODE)
            >>> ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
            >>> init()
            >>> ms.set_seed(1)
            >>> class Network(nn.Cell):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.flatten = nn.Flatten()
            ...         self.fc = nn.Dense(28*28, 10, weight_init="normal", bias_init="zeros")
            ...         self.relu = nn.ReLU()
            ...
            ...     def construct(self, x):
            ...         x = self.flatten(x)
            ...         logits = self.relu(self.fc(x))
            ...         return logits
            >>>
            >>> net = Network()
            >>>
            >>> def create_dataset(batch_size):
            ...     dataset_path = os.getenv("DATA_PATH")
            ...     rank_id = get_rank()
            ...     rank_size = get_group_size()
            ...     dataset = ds.MnistDataset(dataset_path, num_shards=rank_size, shard_id=rank_id)
            ...     image_transforms = [
            ...         ds.vision.Rescale(1.0 / 255.0, 0),
            ...         ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
            ...         ds.vision.HWC2CHW()
            ...     ]
            ...     label_transform = ds.transforms.TypeCast(ms.int32)
            ...     dataset = dataset.map(image_transforms, 'image')
            ...     dataset = dataset.map(label_transform, 'label')
            ...     dataset = dataset.batch(batch_size)
            ...     return dataset
            >>> data_set = create_dataset(32)
            >>> ckpt_file= "./rank_5/iteration-1_40.ckpt"
            >>> strategy_file = "./src_pipeline_strategys/src_strategy_5.ckpt"
            >>> param_dict = MindIOTTPAdapter.load_checkpoint_with_backup(ckpt_file, stragegy_file, net)
            >>> data_set.set_init_step(param_dict["global_step"])
        """
        logger.info("Start load checkpoint with strategy file.")
        try:
            param_dict = ms.load_checkpoint(ckpt_file_path)
        except ValueError as e:
            logger.warning(
                "Loading origin checkpoint file failed, the reason is:{}.".format(str(e)))
            dp = _get_dp_from_layout(strategy_file_path)
            rank = get_rank()
            logger.info(
                "Can't load origin checkpoint file, found dp:{}.".format(dp))
            for i in dp:
                if i == rank:
                    continue
                new_ckpt = ckpt_file_path.replace(
                    f"/rank_{rank}/", f"/rank_{str(i)}/")
                if not os.path.isfile(new_ckpt):
                    continue
                try:
                    param_dict = ms.load_checkpoint(new_ckpt)
                except ValueError as e1:
                    logger.warning(
                        "Loading strategy checkpoint file failed, the reason is:{}.".format(str(e1)))
                    param_dict = None
        if param_dict:
            logger.info("Found param dict, load it into network.")
            ms.load_param_into_net(net, param_dict)
        else:
            raise ValueError(
                "Load checkpoint file failed, please check your config is set correctly.")
        logger.info("Finish load checkpoint with strategy file.")
        return param_dict
