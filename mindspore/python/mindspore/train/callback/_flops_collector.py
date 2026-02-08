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
"""FLops Utilization collector Callback class."""
from __future__ import absolute_import

import time
import os
import stat
import hashlib

from math import floor
from mindspore import _checkparam as Validator
from mindspore import log as logger
from mindspore.train.callback._callback import Callback
from mindspore.graph.api import flops_collection

from mindspore.communication.management import (create_group, get_group_size,
                                                get_rank)
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.ops import operations as P
from mindspore.common import Tensor
import mindspore.nn as nn


class AllReduceNet(nn.Cell):
    """
    Used to accumulate flops in pipeline parallel.
    """
    def __init__(self, group_name):
        super(AllReduceNet, self).__init__()
        self.allreduce_sum = P.AllReduce(op=P.ReduceOp.SUM, group=group_name)
        self.add_flags(skip_auto_parallel_compile=True)

    def construct(self, x):
        return self.allreduce_sum(x)


class FlopsUtilizationCollector(Callback):
    """
    The FlopsUtilizationCollector interface counts the model utilization information MFU
    and the hardware utilization information HFU.
    Currently, the API counts only the forward and backward flops of MatMul,
    BatchMatMul, flash_attention_score, and Conv2D operators.
    Only used in graph mode with static shape.

    Args:
        data_size (int): How many steps are the intervals between print information each time.
        computility (int, optional): The peak flops of each compute card. Default: ``1`` .
        full_flops(bool, optional): Whether to count the full model flops. If set full_flops to False,
            FlopsUtilizationCollector would count the shard model flops in each device. Default: ``True`` .
        enable_ma_collector(bool, optional): Whether to write flops into the log and provide them to tasks
            on the cloud for retrieval. Default: ``False`` .

    Raises:
        TypeError: If data_size is not positive int.
        TypeError: If full_flops is not bool.
        TypeError: If enable_ma_collector is not bool.
        AssertionError: If the training mode is not a static graph or not a static shape.

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> from mindspore import nn
        >>> from mindspore.train import Model, FlopsUtilizationCollector
        >>> from mindspore import context
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
        >>> train_dataset = ds.NumpySlicesDataset(data=data).batch(32)
        >>> net = nn.Dense(10, 5)
        >>> crit = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> flops_callback = FlopsUtilizationCollector(train_dataset.get_dataset_size(), computility=10e6)
        >>> model = Model(network=net, optimizer=opt, loss_fn=crit, metrics={"recall"})
        >>> model.train(2, train_dataset, callbacks=[flops_callback])
        Full model flops is 6400, Full hardware flops is 6400, Shard model flops is 6400, Shard hardware flops is 6400
        Train per step time: 135.572 ms, mfu:0.47% hfu:0.47%
        Train per step time: 1.317 ms, mfu:48.59% hfu:48.59%
    """
    def __init__(self, data_size, computility=1, full_flops=True, enable_ma_collector=False):
        super(FlopsUtilizationCollector, self).__init__()
        self.step_time = time.time()
        self.computility = computility
        self.full_mfu = 0.0
        self.full_hfu = 0.0
        self.shard_mfu = 0.0
        self.shard_hfu = 0.0
        self.full_model_flops = 0.0
        self.shard_model_flops = 0.0
        self.full_hardware_flops = 0.0
        self.shard_hardware_flops = 0.0
        self.mfu_calculated = False
        self.data_size = data_size
        self.time_step_path = ''
        self.full_flops = full_flops
        self.verbose = not(computility == 1 and enable_ma_collector)
        self.ma = enable_ma_collector
        self.batch_step_size = None
        Validator.check_bool(full_flops, "full_flops")
        Validator.check_bool(enable_ma_collector, "enable_ma_collector")
        Validator.check_positive_int(data_size, "data_size")

    def step_begin(self, run_context):
        """
        Record time at the beginning of step.


        Args:
            run_context (RunContext): Context of the process running. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        if self.batch_step_size is None:
            self.batch_step_size = self.data_size
            cb_params = run_context.original_args()
            if hasattr(cb_params, "batch_num"):
                batch_num = cb_params.batch_num
                if isinstance(batch_num, int) and batch_num > 0:
                    self.batch_step_size = cb_params.batch_num
            Validator.check_positive_int(self.batch_step_size)
        self.step_time = time.time()

    def _get_pipeline_group(self):
        """
        Calculate the communication group between all pipeline stages
        """
        rank = get_rank()
        stage_nums = auto_parallel_context().get_pipeline_stages()
        device_nums = get_group_size()
        per_stage_device_nums = device_nums // stage_nums
        local_stage_rank_id = rank % per_stage_device_nums
        group = range(0, stage_nums)
        rank_list = [local_stage_rank_id + x *
                     per_stage_device_nums for x in group]
        rank_str_list = [str(local_stage_rank_id + x *
                             per_stage_device_nums) for x in group]
        rank_list_str = "-".join(rank_str_list)
        return rank_list, rank_list_str

    def _check_run_mode_valid(self, run_context):
        """
        Check whether FlopsUtilizationCollector is working in the current environment
        """
        cb_params = run_context.original_args()
        if cb_params.mode == 'train':
            network = cb_params.train_network
            if not network.compiled:
                if self.verbose:
                    raise ValueError("FlopsUtilizationCollector now only support graph mode.")
                logger.info("FlopsUtilizationCollector now only support graph mode.")
                return False
        elif cb_params.mode == 'eval':
            network = cb_params.eval_network
            if not network.compiled:
                if self.verbose:
                    raise ValueError("FlopsUtilizationCollector now only support graph mode.")
                logger.info("FlopsUtilizationCollector now only support graph mode.")
                return False
        else:
            if self.verbose:
                raise ValueError('FlopsUtilizationCollector only support train and eval mode!')
            logger.info('FlopsUtilizationCollector only support train and eval mode!')
            return False
        try:
            self.full_model_flops, self.full_hardware_flops, self.shard_model_flops, \
            self.shard_hardware_flops, is_dynamic_shape = flops_collection(network.current_phase)
        except Exception as e:
            if self.verbose:
                raise ValueError("FlopsUtilizationCollector is not supported because {}.".format(e))
            logger.info("FlopsUtilizationCollector is not supported because {}.".format(e))
            return False
        if is_dynamic_shape:
            if self.verbose:
                raise ValueError("FlopsUtilizationCollector now do not support dynamic shape.")
            logger.info("FlopsUtilizationCollector now do not support dynamic shape.")
            return False
        return True

    def step_end(self, run_context):
        """
        Print mfu and hfu time at the end of step.

        Args:
           run_context (RunContext): Context of the process running. For more details,
                   please refer to :class:`mindspore.train.RunContext`.
        """
        step_seconds = (time.time() - self.step_time) * 1000
        if not self.mfu_calculated:
            if not self._check_run_mode_valid(run_context):
                return
            self.full_mfu = self.full_model_flops / self.computility
            self.full_hfu = self.full_hardware_flops / self.computility
            self.shard_mfu = self.shard_model_flops / self.computility
            self.shard_hfu = self.shard_hardware_flops / self.computility
            self.mfu_calculated = True
            shard_mf_dir = os.path.realpath(os.getenv('MA_LOG_DIR', './'))
            if self.ma:
                rank_id = get_rank() if auto_parallel_context().get_parallel_mode() != "stand_alone" else 0
                flops_path = os.path.join(
                    shard_mf_dir, "flops_rank_" + str(rank_id)) + ".txt"
                self.time_step_path = os.path.join(
                    shard_mf_dir, "time_step_rank_" + str(rank_id)) + ".txt"
                time_stamp = time.time()
                model_flops_log = "flops{{type=\"model_flops\", rank_id=\"{}\"}} {} {}\n".\
                    format(str(rank_id), self.shard_model_flops, int(round(time_stamp * 1000)))
                hardware_flops_log = "flops{{type=\"hardware_flops\", rank_id=\"{}\"}} {} {}\n".\
                    format(str(rank_id), self.shard_hardware_flops, int(round(time_stamp * 1000)))
                flags = os.O_WRONLY | os.O_CREAT
                modes = stat.S_IWUSR | stat.S_IRUSR
                with os.fdopen(os.open(flops_path, flags, modes), 'w') as f:
                    f.write(model_flops_log)
                    f.write(hardware_flops_log)
            if self.verbose:
                if self.full_flops:
                    pipeline_num = auto_parallel_context().get_pipeline_stages()
                    if pipeline_num > 1:
                        pipeline_group_list, pipeline_group_name = self._get_pipeline_group()
                        auto_parallel_context().set_pipeline_stages(1)
                        hashed = hashlib.md5(
                            pipeline_group_name.encode()).hexdigest()[:48]
                        pipeline_group_name = str(hashed)
                        create_group(pipeline_group_name, pipeline_group_list)
                        self.full_mfu = AllReduceNet(pipeline_group_name)(
                            Tensor([self.full_mfu])).asnumpy()[0]
                        self.full_hfu = AllReduceNet(pipeline_group_name)(
                            Tensor([self.full_hfu])).asnumpy()[0]
                        auto_parallel_context().set_pipeline_stages(pipeline_num)
                    full_model_flops = self.full_mfu * self.computility
                    full_hardware_flops = self.full_hfu * self.computility
                    if auto_parallel_context().get_parallel_mode() != "stand_alone":
                        self.full_mfu = self.full_mfu / get_group_size()
                        self.full_hfu = self.full_hfu / get_group_size()
                    flops_log = f"Full model flops is {full_model_flops}, " \
                                f"Full hardware flops is {full_hardware_flops}, " \
                                f"Shard model flops is {self.shard_model_flops}, " \
                                f"Shard hardware flops is {self.shard_hardware_flops}."
                else:
                    flops_log = f"Shard model flops is {self.shard_model_flops}, " \
                                f"Shard hardware flops is {self.shard_hardware_flops}."
                print(flops_log, flush=True)
        cb_params = run_context.original_args()
        if cb_params.dataset_sink_mode:
            step_seconds = step_seconds / self.batch_step_size
        time_stamp = time.time()
        rank_id = get_rank() if auto_parallel_context().get_parallel_mode() != "stand_alone" else 0
        train_log = "time_monitor{{type=\"per_step_time\", rank_id=\"{}\"}} {} {}".format(
            str(rank_id), step_seconds, int(round(time_stamp * 1000)))
        if self.ma:
            flags = os.O_WRONLY | os.O_CREAT
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(self.time_step_path, flags, modes), 'w') as f:
                f.write(train_log + '\n')
        train_log = "{} per step time: {:5.3f} ms".format(
            cb_params.mode.title(), step_seconds)
        if self.verbose and cb_params.cur_step_num % self.data_size:
            if self.full_flops:
                mfu = 1000 * self.full_mfu / step_seconds
                hfu = 1000 * self.full_hfu / step_seconds
            else:
                mfu = 1000 * self.shard_mfu / step_seconds
                hfu = 1000 * self.shard_hfu / step_seconds

            def floored_percentage(index, val, digits):
                val *= 10 ** (digits + 2)
                return index + '{1:.{0}f}%'.format(digits, floor(val) / 10 ** digits)
            train_log += floored_percentage(' mfu:', mfu, 2)
            train_log += floored_percentage(' hfu:', hfu, 2)
            print(train_log, flush=True)
