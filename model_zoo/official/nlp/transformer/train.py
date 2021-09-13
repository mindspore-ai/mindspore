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
"""Transformer training script."""

import os
import time

from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import Callback, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.communication.management as D
from mindspore.communication.management import get_rank
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.common import set_seed

from src.transformer_for_train import TransformerTrainOneStepCell, TransformerNetworkWithLoss, \
                                      TransformerTrainOneStepWithLossScaleCell, \
                                      TransformerTrainAccumulationAllReducePostWithLossScaleCell
from src.dataset import create_transformer_dataset
from src.lr_schedule import create_dynamic_lr
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id


set_seed(1)

def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))

time_stamp_init = False
time_stamp_first = 0

config.dtype = mstype.float32
config.compute_type = mstype.float16
config.lr_schedule = edict({
    'learning_rate': 2.0,
    'warmup_steps': 8000,
    'start_decay_step': 16000,
    'min_lr': 0.0,
    })

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_id=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_id
        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = get_ms_timestamp()
            time_stamp_init = True

    def step_end(self, run_context):
        """Monitor the loss in training."""
        global time_stamp_first
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - time_stamp_first,
                                                                     cb_params.cur_epoch_num,
                                                                     cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))

        loss_file = "./loss_{}.log"
        if config.enable_modelarts:
            loss_file = "/cache/train/loss_{}.log"

        with open(loss_file.format(self.rank_id), "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, loss: {}, overflow: {}, loss_scale: {}".format(
                time_stamp_current - time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                str(cb_params.net_outputs[0].asnumpy()),
                str(cb_params.net_outputs[1].asnumpy()),
                str(cb_params.net_outputs[2].asnumpy())))
            f.write('\n')


def modelarts_pre_process():
    config.save_checkpoint_path = config.output_path
    config.data_path = os.path.join(config.data_path, 'ende-l128-mindrecord')

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_transformer_train():
    """
    Transformer training.
    """
    if config.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    context.set_context(reserve_class_name_in_scope=False)

    if config.device_target == "GPU":
        # Enable graph kernel
        context.set_context(enable_graph_kernel=True, graph_kernel_flags="--enable_parallel_fusion")
    if config.distribute == "true":
        if config.device_target == "Ascend":
            device_num = config.device_num
            D.init('hccl')
        else:
            D.init('nccl')
            device_num = D.get_group_size()
            rank = get_rank()
            config.device_id = rank
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        rank_id = config.device_id % device_num
        save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(get_rank()) + '/')
    else:
        device_num = 1
        rank_id = 0
        save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_0/')
    dataset = create_transformer_dataset(epoch_count=1, rank_size=device_num,
                                         rank_id=rank_id, do_shuffle=config.do_shuffle,
                                         dataset_path=config.data_path,
                                         bucket_boundaries=config.bucket_boundaries,
                                         device_target=config.device_target)

    netwithloss = TransformerNetworkWithLoss(config, True)

    if config.checkpoint_path:
        parameter_dict = load_checkpoint(config.checkpoint_path)
        load_param_into_net(netwithloss, parameter_dict)

    hidden_size = config.hidden_size
    learning_rate = config.lr_schedule.learning_rate if config.device_target == "Ascend" else 1.0
    lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                  training_steps=dataset.get_dataset_size()*config.epoch_size,
                                  learning_rate=learning_rate,
                                  warmup_steps=config.lr_schedule.warmup_steps,
                                  hidden_size=hidden_size,
                                  start_decay_step=config.lr_schedule.start_decay_step,
                                  min_lr=config.lr_schedule.min_lr), mstype.float32)

    if config.device_target == "GPU" and config.transformer_network == "large":
        optimizer = Adam(netwithloss.trainable_params(), lr, beta2=config.optimizer_adam_beta2)
    else:
        optimizer = Adam(netwithloss.trainable_params(), lr)

    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(rank_id=rank_id)]
    if config.enable_save_ckpt == "true":
        if device_num == 1 or (device_num > 1 and rank_id == 0):
            if config.device_target == "Ascend":
                ckpt_config = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                               keep_checkpoint_max=config.save_checkpoint_num)
            else:
                ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset.get_dataset_size(),
                                               keep_checkpoint_max=config.save_checkpoint_num)
            ckpoint_cb = ModelCheckpoint(prefix='transformer', directory=save_ckpt_path, config=ckpt_config)
            callbacks.append(ckpoint_cb)

    if config.enable_lossscale == "true":
        scale_manager = DynamicLossScaleManager(init_loss_scale=config.init_loss_scale_value,
                                                scale_factor=config.scale_factor,
                                                scale_window=config.scale_window)
        update_cell = scale_manager.get_update_cell()
        if config.accumulation_steps > 1:
            netwithgrads = TransformerTrainAccumulationAllReducePostWithLossScaleCell(netwithloss, optimizer,
                                                                                      update_cell,
                                                                                      config.accumulation_steps)
        else:
            netwithgrads = TransformerTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer,
                                                                    scale_update_cell=update_cell)
    else:
        netwithgrads = TransformerTrainOneStepCell(netwithloss, optimizer=optimizer)

    netwithgrads.set_train(True)
    model = Model(netwithgrads)

    model.train(config.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=False)


if __name__ == '__main__':
    run_transformer_train()
