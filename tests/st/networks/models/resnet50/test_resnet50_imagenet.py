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

"""train and evaluate resnet50 network on imagenet dataset"""

import os
import time
import numpy as np

from mindspore import context
from mindspore.common.tensor import Tensor
import mindspore.communication.management as D
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train import Callback
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import mindspore.nn as nn
import mindspore.dataset as ds

from tests.st.networks.models.resnet50.src.resnet import resnet50
from tests.st.networks.models.resnet50.src.dataset import create_dataset
from tests.st.networks.models.resnet50.src.lr_generator import get_learning_rate
from tests.st.networks.models.resnet50.src.config import config
from tests.st.networks.models.resnet50.src.metric import DistAccuracy, ClassifyCorrectCell
from tests.st.networks.models.resnet50.src.CrossEntropySmooth import CrossEntropySmooth


DATASET_PATH = "/home/workspace/mindspore_dataset/imagenet/imagenet_original/train"
EVAL_PATH = "/home/workspace/mindspore_dataset/imagenet/imagenet_original/val"

np.random.seed(1)
ds.config.set_seed(1)
os.environ['GLOG_v'] = str(2)


class LossGet(Callback):
    def __init__(self, per_print_times, data_size):
        super(LossGet, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self._loss = 0.0
        self.data_size = data_size
        self.epoch_time = None
        self._per_step_mseconds = None

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training."
                             .format(cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            self._loss = loss

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        self._per_step_mseconds = epoch_mseconds / self.data_size

    def get_loss(self):
        return self._loss

    def get_per_step_time(self):
        return self._per_step_mseconds


def test_resnet_imagenet_8p_mpi():
    """
    Feature: Resnet50 network.
    Description: Train and evaluate resnet50 network on imagenet dataset with mpi.
    Expectation: accuracy > 0.1, time cost < 26.
    """
    epoch_size = 2
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(enable_graph_kernel=False)
    context.reset_auto_parallel_context()
    context.reset_ps_context()
    context.set_auto_parallel_context(device_num=8, parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True, all_reduce_fusion_config=[107, 160])
    D.init()

    # network
    net = resnet50(class_num=config.class_num)

    # evaluation network
    dist_eval_network = ClassifyCorrectCell(net)

    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0

    # loss
    loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=config.label_smooth_factor,
                              num_classes=config.class_num)

    # train dataset
    dataset = create_dataset(dataset_path=DATASET_PATH, do_train=True,
                             repeat_num=1, batch_size=config.batch_size)

    step_size = dataset.get_dataset_size()
    eval_interval = config.eval_interval
    dataset.__loop_size__ = step_size * eval_interval

    # evaluation dataset
    eval_dataset = create_dataset(dataset_path=EVAL_PATH, do_train=False,
                                  repeat_num=1, batch_size=config.eval_batch_size)

    # loss scale
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    # learning rate
    lr = Tensor(get_learning_rate(lr_init=config.lr_init, lr_end=0.0, lr_max=config.lr_max,
                                  warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size,
                                  steps_per_epoch=step_size, lr_decay_mode=config.lr_decay_mode))

    # optimizer
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params, 'weight_decay': 0.0},
                    {'order_params': net.trainable_params()}]

    if config.use_lars:
        momentum = nn.Momentum(group_params, lr, config.momentum,
                               loss_scale=config.loss_scale, use_nesterov=config.use_nesterov)
        opt = nn.LARS(momentum, epsilon=config.lars_epsilon, coefficient=config.lars_coefficient,
                      lars_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'bias' not in x.name)

    else:
        opt = nn.Momentum(group_params, lr, config.momentum,
                          loss_scale=config.loss_scale, use_nesterov=config.use_nesterov)

    # model
    model = Model(net, loss_fn=loss, optimizer=opt,
                  loss_scale_manager=loss_scale, amp_level="O2", keep_batchnorm_fp32=False,
                  metrics={'acc': DistAccuracy(batch_size=config.eval_batch_size, device_num=8)},
                  eval_network=dist_eval_network)

    # callbacks
    loss_cb = LossGet(1, step_size)

    # train and eval
    acc = 0.0
    time_cost = 0.0
    for _ in range(0, int(epoch_size / eval_interval)):
        model.train(1, dataset, callbacks=loss_cb, dataset_sink_mode=True)
        output = model.eval(eval_dataset)
        acc = float(output.get('acc', 0.0))
        time_cost = loss_cb.get_per_step_time()
    assert acc > 0.1
    assert time_cost < 26
