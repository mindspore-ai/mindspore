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

from mindspore import context
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.train.model import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import get_rank, init, get_group_size

from src.models import Monitor


def switch_precision(net, data_type, config):
    if config.platform == "Ascend":
        net.to_float(data_type)
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.to_float(mstype.float32)


def context_device_init(config):
    if config.platform == "CPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, save_graphs=False)

    elif config.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, save_graphs=False)
        if config.run_distribute:
            init("nccl")
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)

    elif config.platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, device_id=config.device_id,
                            save_graphs=False)
        if config.run_distribute:
            context.set_auto_parallel_context(device_num=config.rank_size,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    else:
        raise ValueError("Only support CPU, GPU and Ascend.")


def set_context(config):
    if config.platform == "CPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform,
                            save_graphs=False)
    elif config.platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform,
                            device_id=config.device_id, save_graphs=False)
    elif config.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=config.platform, save_graphs=False)


def config_ckpoint(config, lr, step_size):
    cb = None
    if config.platform in ("CPU", "GPU") or config.rank_id == 0:
        cb = [Monitor(lr_init=lr.asnumpy())]

        if config.save_checkpoint:
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                         keep_checkpoint_max=config.keep_checkpoint_max)

            rank = 0
            if config.run_distribute:
                rank = get_rank()

            ckpt_save_dir = config.save_checkpoint_path + "ckpt_" + str(rank) + "/"
            ckpt_cb = ModelCheckpoint(prefix="mobilenetv2", directory=ckpt_save_dir, config=config_ck)
            cb += [ckpt_cb]
    return cb
