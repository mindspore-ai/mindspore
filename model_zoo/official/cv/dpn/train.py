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
"""DPN model train with MindSpore"""
import os
from ast import literal_eval
from mindspore import context
from mindspore import Tensor
from mindspore.nn import SGD
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init, get_group_size
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.imagenet_dataset import classification_dataset
from src.dpn import dpns
from src.lr_scheduler import get_lr_drop, get_lr_warmup
from src.crossentropy import CrossEntropy
from src.callbacks import SaveCallback
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_rank_id, get_device_num


set_seed(1)


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def dpn_train():
    # init context
    device_id = get_device_id()
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target, save_graphs=False, device_id=device_id)
    # init distributed
    if config.is_distributed:
        init()
        config.rank = get_rank_id()
        config.group_size = get_group_size()
        config.device_num = get_device_num()
        context.set_auto_parallel_context(device_num=config.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1
    # create dataset
    train_dataset = classification_dataset(config.train_data_dir,
                                           image_size=config.image_size,
                                           per_batch_size=config.batch_size,
                                           max_epoch=1,
                                           num_parallel_workers=config.num_parallel_workers,
                                           shuffle=True,
                                           rank=config.rank,
                                           group_size=config.group_size)
    if config.eval_each_epoch:
        print("create eval_dataset")
        eval_dataset = classification_dataset(config.eval_data_dir,
                                              image_size=config.image_size,
                                              per_batch_size=config.batch_size,
                                              max_epoch=1,
                                              num_parallel_workers=config.num_parallel_workers,
                                              shuffle=False,
                                              rank=config.rank,
                                              group_size=config.group_size,
                                              mode='eval')
    train_step_size = train_dataset.get_dataset_size()

    # choose net
    net = dpns[config.backbone](num_classes=config.num_classes)

    # load checkpoint
    if os.path.isfile(config.pretrained):
        print("load ckpt")
        load_param_into_net(net, load_checkpoint(config.pretrained))
    # learing rate schedule
    if config.lr_schedule == 'drop':
        print("lr_schedule:drop")
        lr = Tensor(get_lr_drop(global_step=config.global_step,
                                total_epochs=config.epoch_size,
                                steps_per_epoch=train_step_size,
                                lr_init=config.lr_init,
                                factor=config.factor))
    elif config.lr_schedule == 'warmup':
        print("lr_schedule:warmup")
        lr = Tensor(get_lr_warmup(global_step=config.global_step,
                                  total_epochs=config.epoch_size,
                                  steps_per_epoch=train_step_size,
                                  lr_init=config.lr_init,
                                  lr_max=config.lr_max,
                                  warmup_epochs=config.warmup_epochs))

    # optimizer
    config.weight_decay = literal_eval(config.weight_decay)
    opt = SGD(net.trainable_params(),
              lr,
              momentum=config.momentum,
              weight_decay=config.weight_decay,
              loss_scale=config.loss_scale_num)
    # loss scale
    loss_scale = FixedLossScaleManager(config.loss_scale_num, False)
    # loss function
    if config.dataset == "imagenet-1K":
        print("Use SoftmaxCrossEntropyWithLogits")
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    else:
        if not config.label_smooth:
            config.label_smooth_factor = 0.0
        print("Use Label_smooth CrossEntropy")
        loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)
    # create model
    model = Model(net, amp_level="O2",
                  keep_batchnorm_fp32=False,
                  loss_fn=loss,
                  optimizer=opt,
                  loss_scale_manager=loss_scale,
                  metrics={'top_1_accuracy', 'top_5_accuracy'})

    # loss/time monitor & ckpt save callback
    loss_cb = LossMonitor()
    time_cb = TimeMonitor(data_size=train_step_size)
    cb = [loss_cb, time_cb]
    if config.rank_save_ckpt_flag:
        if config.eval_each_epoch:
            save_cb = SaveCallback(model, eval_dataset, config.ckpt_path)
            cb += [save_cb]
        else:
            config_ck = CheckpointConfig(save_checkpoint_steps=train_step_size,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            ckpoint_cb = ModelCheckpoint(prefix="dpn", directory=config.ckpt_path, config=config_ck)
            cb.append(ckpoint_cb)
    # train model
    model.train(config.epoch_size, train_dataset, callbacks=cb)


if __name__ == '__main__':
    dpn_train()
    print('DPN training success!')
