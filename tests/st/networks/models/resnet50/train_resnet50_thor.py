# Copyright 2022 Huawei Technologies Co., Ltd
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
"""resnet train & eval case."""
import os
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.train import ConvertModelUtils
from tests.st.networks.models.resnet50.src.callback import LossGet
from tests.st.networks.models.resnet50.src_thor.config import config as thor_config
from tests.st.networks.models.resnet50.src_thor.dataset import create_dataset2 as create_dataset_thor
from tests.st.networks.models.resnet50.src.resnet import resnet50
from tests.st.networks.models.resnet50.src.metric import DistAccuracy, ClassifyCorrectCell
from tests.st.networks.models.resnet50.src.CrossEntropySmooth import CrossEntropySmooth

TRAIN_PATH = "/home/workspace/mindspore_dataset/imagenet/imagenet_original/train"
EVAL_PATH = "/home/workspace/mindspore_dataset/imagenet/imagenet_original/val"
ms.set_seed(1)


def get_thor_lr(global_step, lr_init, decay, total_epochs, steps_per_epoch, decay_epochs=100):
    """get_model_lr"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for i in range(total_steps):
        epoch = (i + 1) / steps_per_epoch
        base = (1.0 - float(epoch) / total_epochs) ** decay
        lr_local = lr_init * base
        if epoch >= decay_epochs:
            lr_local = lr_local * 0.5
        if epoch >= decay_epochs + 1:
            lr_local = lr_local * 0.5
        lr_each_step.append(lr_local)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate


def get_thor_damping(global_step, damping_init, decay_rate, total_epochs, steps_per_epoch):
    """get_model_damping"""
    damping_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for step in range(total_steps):
        epoch = (step + 1) / steps_per_epoch
        damping_here = damping_init * (decay_rate ** (epoch / 10))
        damping_each_step.append(damping_here)
    current_step = global_step
    damping_each_step = np.array(damping_each_step).astype(np.float32)
    damping_now = damping_each_step[current_step:]
    return damping_now


def run_train():
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
    rank_id = int(os.getenv('RANK_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))
    device_id = int(os.getenv('DEVICE_ID', '0'))
    print(f"run resnet50 thor device_num:{device_num}, device_id:{device_id}, rank_id:{rank_id}")
    if device_num > 1:
        ms.communication.init()
        ms.context.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                             gradients_mean=True, all_reduce_fusion_config=[80, 160])
    net = resnet50(thor_config.class_num)

    if not thor_config.label_smooth:
        thor_config.label_smooth_factor = 0.0

    # loss
    loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=thor_config.label_smooth_factor,
                              num_classes=thor_config.class_num)

    # train dataset
    dataset = create_dataset_thor(dataset_path=TRAIN_PATH, do_train=True,
                                  batch_size=thor_config.batch_size, train_image_size=thor_config.train_image_size,
                                  eval_image_size=thor_config.eval_image_size, target="Ascend",
                                  distribute=True)
    step_size = dataset.get_dataset_size()

    # loss scale
    loss_scale = ms.FixedLossScaleManager(thor_config.loss_scale, drop_overflow_update=False)

    # learning rate
    lr = get_thor_lr(0, 0.05803, 4.04839, 53, 5004, decay_epochs=39)
    damping = get_thor_damping(0, 0.02714, 0.50036, 70, 5004)
    # optimizer
    split_indices = [26, 53]
    opt = nn.thor(net, ms.Tensor(lr), ms.Tensor(damping), thor_config.momentum, thor_config.weight_decay,
                  thor_config.loss_scale, thor_config.batch_size, split_indices=split_indices,
                  frequency=thor_config.frequency)

    # evaluation network
    dist_eval_network = ClassifyCorrectCell(net)
    # model
    model = ms.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale,
                     metrics={'acc': DistAccuracy(batch_size=thor_config.eval_batch_size, device_num=device_num)},
                     amp_level="O2", keep_batchnorm_fp32=False,
                     eval_network=dist_eval_network)

    model = ConvertModelUtils().convert_to_thor_model(model=model, network=net, loss_fn=loss, optimizer=opt,
                                                      loss_scale_manager=loss_scale, metrics={'acc'},
                                                      amp_level="O2", keep_batchnorm_fp32=False)

    # callbacks
    loss_cb = LossGet(1, step_size)

    # train and eval
    print("run_start", device_id)
    model.train(2, dataset, callbacks=loss_cb, dataset_sink_mode=True, sink_size=step_size)
    time_cost = loss_cb.get_per_step_time()
    loss = loss_cb.get_loss()
    epoch_idx = loss_cb.get_epoch()
    print("the {} epoch's resnet result:\n "
          "device{}, training loss {}, "
          "training per step cost {:.2f} ms, total_cost {:.2f} ms".format(epoch_idx, device_id,
                                                                          loss, time_cost, time_cost * step_size))
    print(f"#-#resnet_thor_loss: {loss}")
    print(f"#-#resnet_thor_time_cost: {time_cost}")

if __name__ == '__main__':
    run_train()
