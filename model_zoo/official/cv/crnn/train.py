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
"""crnn training"""
import os
import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.nn.wrap import WithLossCell
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_group_size, get_rank
from src.loss import CTCLoss
from src.dataset import create_dataset
from src.crnn import crnn
from src.crnn_for_train import TrainOneStepCellWithGradClip
from src.metric import CRNNAccuracy
from src.eval_callback import EvalCallBack
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.model_utils.device_adapter import get_rank_id, get_device_num, get_device_id

set_seed(1)

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)


def apply_eval(eval_param):
    evaluation_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = evaluation_model.eval(eval_ds)
    return res[metrics_name]


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    if config.device_target == 'Ascend':
        device_id = get_device_id()
        context.set_context(device_id=device_id)

    # lr_scale = 1
    if config.run_distribute:
        if config.device_target == 'Ascend':
            init()
            # lr_scale = 1
            device_num = get_device_num()
            rank = get_rank_id()
        else:
            init()
            # lr_scale = 1
            device_num = get_group_size()
            rank = get_rank()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        device_num = 1
        rank = 0

    max_text_length = config.max_text_length
    # create dataset
    dataset = create_dataset(name=config.train_dataset, dataset_path=config.train_dataset_path,
                             batch_size=config.batch_size,
                             num_shards=device_num, shard_id=rank, config=config)
    step_size = dataset.get_dataset_size()
    print("step_size:", step_size)
    # define lr
    lr_init = config.learning_rate
    lr = nn.dynamic_lr.cosine_decay_lr(0.0, lr_init, config.epoch_size * step_size, step_size, config.epoch_size)
    loss = CTCLoss(max_sequence_length=config.num_step,
                   max_label_length=max_text_length,
                   batch_size=config.batch_size)
    net = crnn(config, full_precision=config.device_target == 'GPU')
    opt = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum, nesterov=config.nesterov)

    net_with_loss = WithLossCell(net, loss)
    net_with_grads = TrainOneStepCellWithGradClip(net_with_loss, opt).set_train()
    # define model
    model = Model(net_with_grads)
    # define callbacks
    callbacks = [LossMonitor(per_print_times=config.per_print_time),
                 TimeMonitor(data_size=step_size)]
    save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(rank) + '/')
    if config.run_eval and rank == 0:
        if config.train_eval_dataset_path is None or (not os.path.isdir(config.train_eval_dataset_path)):
            raise ValueError("{} is not a existing path.".format(config.train_eval_dataset_path))
        eval_dataset = create_dataset(name=config.train_eval_dataset,
                                      dataset_path=config.train_eval_dataset_path,
                                      batch_size=config.batch_size,
                                      is_training=False,
                                      config=config)
        eval_model = Model(net, loss, metrics={'CRNNAccuracy': CRNNAccuracy(config, print_flag=False)})
        eval_param_dict = {"model": eval_model, "dataset": eval_dataset, "metrics_name": "CRNNAccuracy"}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=config.eval_start_epoch, save_best_ckpt=True,
                               ckpt_directory=save_ckpt_path, best_ckpt_name="best_acc.ckpt",
                               eval_all_saved_ckpts=config.eval_all_saved_ckpts, metrics_name="acc")
        callbacks += [eval_cb]
    if config.save_checkpoint and rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="crnn", directory=save_ckpt_path, config=config_ck)
        callbacks.append(ckpt_cb)
    model.train(config.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=config.device_target == 'Ascend')


if __name__ == '__main__':
    train()
