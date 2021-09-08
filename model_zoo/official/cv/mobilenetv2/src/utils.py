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
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import get_rank, init, get_group_size

from src.models import Monitor

def context_device_init(config):
    if config.platform == "GPU" and config.run_distribute:
        config.device_id = 0
    config.rank_id = 0
    config.rank_size = 1
    if config.platform == "CPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, save_graphs=False)

    elif config.platform in ["Ascend", "GPU"]:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, device_id=config.device_id,
                            save_graphs=False)
        if config.run_distribute:
            init()
            config.rank_id = get_rank()
            config.rank_size = get_group_size()
            context.set_auto_parallel_context(device_num=config.rank_size,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        raise ValueError("Only support CPU, GPU and Ascend.")


def config_ckpoint(config, lr, step_size, model=None, eval_dataset=None):
    cb = [Monitor(lr_init=lr.asnumpy(), model=model, eval_dataset=eval_dataset)]
    if config.save_checkpoint and config.rank_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_save_dir = config.save_checkpoint_path + "ckpt_" + str(config.rank_id) + "/"
        ckpt_cb = ModelCheckpoint(prefix="mobilenetv2", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]
    return cb
