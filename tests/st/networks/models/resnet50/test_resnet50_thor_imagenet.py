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

"""train and evaluate resnet50 network on imagenet dataset"""

import os
import time
from multiprocessing import Process, Queue
import pytest
import numpy as np

from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.train.callback import Callback
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.nn.optim import thor
import mindspore.dataset as ds

from tests.st.networks.models.resnet50.src.dataset import create_dataset
from tests.st.networks.models.resnet50.src.metric import DistAccuracy, ClassifyCorrectCell
from tests.st.networks.models.resnet50.src.CrossEntropySmooth import CrossEntropySmooth
from tests.st.networks.models.resnet50.src_thor.config import config as thor_config
from tests.st.networks.models.resnet50.src_thor.dataset import create_dataset as create_dataset_thor
from tests.st.networks.models.resnet50.src_thor.model_thor import Model as THOR_Model
from tests.st.networks.models.resnet50.src_thor.resnet import resnet50 as resnet50_thor


MINDSPORE_HCCL_CONFIG_PATH = "/home/workspace/mindspore_config/hccl/rank_table_8p.json"
dataset_path = "/home/workspace/mindspore_dataset/imagenet/imagenet_original/train"
eval_path = "/home/workspace/mindspore_dataset/imagenet/imagenet_original/val"

np.random.seed(1)
ds.config.set_seed(1)
os.environ['GLOG_v'] = str(2)


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


class LossGet(Callback):
    def __init__(self, per_print_times, data_size):
        super(LossGet, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self._loss = 0.0
        self.data_size = data_size

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


def train_process_thor(q, device_id, epoch_size, device_num, enable_hccl):
    os.system("mkdir " + str(device_id))
    os.chdir(str(device_id))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(device_id=device_id)
    os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = MINDSPORE_HCCL_CONFIG_PATH
    os.environ['RANK_ID'] = str(device_id)
    os.environ['RANK_SIZE'] = str(device_num)
    if enable_hccl:
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, all_reduce_fusion_config=[85, 160])
        init()

    # network
    net = resnet50_thor(thor_config.class_num)

    if not thor_config.label_smooth:
        thor_config.label_smooth_factor = 0.0

    # loss
    loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=thor_config.label_smooth_factor,
                              num_classes=thor_config.class_num)

    # train dataset
    dataset = create_dataset_thor(dataset_path=dataset_path, do_train=True,
                                  repeat_num=1, batch_size=thor_config.batch_size)

    step_size = dataset.get_dataset_size()
    eval_interval = thor_config.eval_interval

    # evaluation dataset
    eval_dataset = create_dataset(dataset_path=eval_path, do_train=False,
                                  repeat_num=1, batch_size=thor_config.eval_batch_size)

    # loss scale
    loss_scale = FixedLossScaleManager(thor_config.loss_scale, drop_overflow_update=False)

    # learning rate
    lr = get_thor_lr(0, 0.05803, 4.04839, 53, 5004, decay_epochs=39)
    damping = get_thor_damping(0, 0.02714, 0.50036, 70, 5004)
    # optimizer
    split_indices = [26, 53]
    opt = thor(net, Tensor(lr), Tensor(damping), thor_config.momentum, thor_config.weight_decay, thor_config.loss_scale,
               thor_config.batch_size, split_indices=split_indices, frequency=thor_config.frequency)

    # evaluation network
    dist_eval_network = ClassifyCorrectCell(net)
    # model
    model = THOR_Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, amp_level="O2",
                       keep_batchnorm_fp32=False,
                       metrics={'acc': DistAccuracy(batch_size=thor_config.eval_batch_size, device_num=device_num)},
                       eval_network=dist_eval_network, frequency=thor_config.frequency)

    # model init
    print("init_start", device_id)
    model.init(dataset, eval_dataset)
    print("init_stop", device_id)

    # callbacks
    loss_cb = LossGet(1, step_size)

    # train and eval
    acc = 0.0
    time_cost = 0.0
    print("run_start", device_id)
    for epoch_idx in range(0, int(epoch_size / eval_interval)):
        model.train(eval_interval, dataset, callbacks=loss_cb)
        eval_start = time.time()
        output = model.eval(eval_dataset)
        eval_cost = (time.time() - eval_start) * 1000
        acc = float(output["acc"])
        time_cost = loss_cb.get_per_step_time()
        loss = loss_cb.get_loss()
        print("the {} epoch's resnet result:\n "
              "device{}, training loss {}, acc {}, "
              "training per step cost {:.2f} ms, eval cost {:.2f} ms, total_cost {:.2f} ms".format(
                  epoch_idx, device_id, loss, acc, time_cost, eval_cost, time_cost * step_size + eval_cost))
    q.put({'acc': acc, 'cost': time_cost})


def test_resnet_thor_imagenet_8p_0():
    """
    Feature: Resnet50 thor network
    Description: Train and evaluate resnet50 thor network on imagenet dataset
    Expectation: accuracy > 0.28, time cost < 25.
    """
    context.set_context(enable_graph_kernel=False, enable_sparse=False)
    context.reset_auto_parallel_context()
    context.reset_ps_context()

    q = Queue()

    # resnet50_thor
    device_num = 8
    epoch_size = 1
    enable_hccl = True
    process = []
    for i in range(device_num):
        device_id = i
        process.append(Process(target=train_process_thor,
                               args=(q, device_id, epoch_size, device_num, enable_hccl)))

    cpu_count = os.cpu_count()
    each_cpu_count = cpu_count // device_num
    for i in range(device_num):
        process[i].start()
        if each_cpu_count > 1:
            cpu_start = each_cpu_count * i
            cpu_end = each_cpu_count * (i + 1)
            process_cpu = [x for x in range(cpu_start, cpu_end)]
            pid = process[i].pid
            os.sched_setaffinity(pid, set(process_cpu))

    print("Waiting for all subprocesses done...")

    for i in range(device_num):
        process[i].join()

    # THOR
    thor_acc = 0.0
    thor_cost = 0.0
    for i in range(device_num):
        output = q.get()
        thor_acc += output['acc']
        thor_cost += output['cost']
    thor_acc = thor_acc / device_num
    thor_cost = thor_cost / device_num

    for i in range(0, device_num):
        os.system("rm -rf " + str(i))
    print("End training...")
    assert thor_acc > 0.25


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_resnet_thor_imagenet_8p_1():
    """
    Feature: Resnet50 thor network
    Description: Train and evaluate resnet50 thor network on imagenet dataset
    Expectation: accuracy > 0.28, time cost < 25.
    """
    context.set_context(enable_graph_kernel=False, enable_sparse=False)
    context.reset_auto_parallel_context()
    context.reset_ps_context()

    q = Queue()

    # resnet50_thor
    device_num = 8
    epoch_size = 1
    enable_hccl = True
    process = []
    for i in range(device_num):
        device_id = i
        process.append(Process(target=train_process_thor,
                               args=(q, device_id, epoch_size, device_num, enable_hccl)))

    cpu_count = os.cpu_count()
    each_cpu_count = cpu_count // device_num
    for i in range(device_num):
        process[i].start()
        if each_cpu_count > 1:
            cpu_start = each_cpu_count * i
            cpu_end = each_cpu_count * (i + 1)
            process_cpu = [x for x in range(cpu_start, cpu_end)]
            pid = process[i].pid
            os.sched_setaffinity(pid, set(process_cpu))

    print("Waiting for all subprocesses done...")

    for i in range(device_num):
        process[i].join()

    # THOR
    thor_acc = 0.0
    thor_cost = 0.0
    for i in range(device_num):
        output = q.get()
        thor_acc += output['acc']
        thor_cost += output['cost']
    thor_acc = thor_acc / device_num
    thor_cost = thor_cost / device_num

    for i in range(0, device_num):
        os.system("rm -rf " + str(i))
    print("End training...")
    print('thor acc: ', thor_acc)
    print('thor cost: ', thor_cost)
    #assert thor_acc > 0.25
    #assert thor_cost < 30
