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

"""
GPT train script
"""


import os
import argparse
from mindspore import context
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.common.dtype as mstype
from mindspore.common import set_seed
from src.dataset import create_dataset
from src.gpt import GPT, GPTWithLoss, CrossEntropyLoss
from src.gpt_wrapcell import GPTTrainOneStepWithLossScaleCell
from src.utils import GPTConfig, LearningRate

def run_train():
    """train function for GPT"""
    parser = argparse.ArgumentParser(description="GPT training")
    parser.add_argument('--device_id', type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--distribute", type=str, default="false", choices=["true", "false"],
                        help="Run distribute, default is false.")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "lamb"],
                        help="select which optimizer to be used, default adam")
    parser.add_argument("--epoch_size", type=int, default=10, help="Epoch size, default is 10.")
    parser.add_argument("--warmup_step", type=int, default=10000, help="Warmup step, default is 10000.")
    parser.add_argument("--data_path", type=str, default="", help="Data path of your MindRecord files.")
    parser.add_argument("--start_lr", type=float, default="5e-5", help="Start learning rate, default is 5e-5.")
    parser.add_argument("--end_lr", type=float, default="1e-10", help="End learning rate, default is 1e-10.")
    parser.add_argument("--sink_size", type=int, default=100, help="Sink size for every iteration, default is 100")


    args_opt = parser.parse_args()
    device_id = int(os.getenv("DEVICE_ID"))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    if args_opt.distribute == "true":
        D.init()
        device_num = args_opt.device_num
        rank = device_id % device_num
        print("device_id is {}, rank_id is {}".format(device_id, rank))

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)

    else:
        rank = 0
        device_num = 1

    config = GPTConfig(batch_size=4,
                       seq_length=1024,
                       vocab_size=50257,
                       embedding_size=1024,
                       num_layers=24,
                       num_heads=16,
                       expand_ratio=4,
                       post_layernorm_residual=False,
                       dropout_rate=0.1,
                       compute_dtype=mstype.float16,
                       use_past=False)
    gpt = GPT(config)
    loss = CrossEntropyLoss(config)
    gpt_with_loss = GPTWithLoss(gpt, loss)

    ds = create_dataset(config.batch_size, data_path=args_opt.data_path, device_num=device_num, rank=rank)


    epoch_num = args_opt.epoch_size
    step_per_epoch = ds.get_dataset_size()

    lr = LearningRate(learning_rate=args_opt.start_lr,
                      end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step,
                      decay_steps=epoch_num*step_per_epoch)

    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    params = gpt.trainable_params()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': 1e-2},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]

    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    else:
        optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr)

    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch/callback_size)
    callback = [TimeMonitor(callback_size), LossMonitor(callback_size)]

    config_ck = CheckpointConfig(save_checkpoint_steps=step_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="GPT2", config=config_ck)
    callback.append(ckpoint_cb)


    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=1024,
                                             scale_factor=2,
                                             scale_window=1000)

    gpt_with_grads = GPTTrainOneStepWithLossScaleCell(gpt_with_loss, optimizer=optimizer,
                                                      scale_update_cell=update_cell)


    model = Model(gpt_with_grads)
    model.train(actual_epoch_num, ds, callbacks=callback, sink_size=callback_size)


if __name__ == "__main__":
    set_seed(12315)
    run_train()
