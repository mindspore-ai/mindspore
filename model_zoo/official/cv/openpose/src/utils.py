# Copyright 2020 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import argparse
import os
import time
import numpy as np

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import LossMonitor
from mindspore.common.tensor import Tensor

from src.config import params

class MyLossMonitor(LossMonitor):
    def __init__(self, per_print_times=1):
        super(MyLossMonitor, self).__init__()
        self._per_print_times = per_print_times
        self._start_time = time.time()
        self._loss_list = []

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
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            # print("epoch: %s step: %s, loss is %s, step time: %.3f s." % (cb_params.cur_epoch_num, cur_step_in_epoch,
            #                                                               loss,
            #                                                               (time.time() - self._start_time)), flush=True)
            self._loss_list.append(loss)
            if cb_params.cur_step_num % 100 == 0:
                print("epoch: %s, steps: [%s] mean loss is: %s"%(cb_params.cur_epoch_num, cur_step_in_epoch,
                                                                 np.array(self._loss_list).mean()), flush=True)
                self._loss_list = []

        self._start_time = time.time()


def parse_args():
    """Parse train arguments."""
    parser = argparse.ArgumentParser('mindspore openpose training')

    # dataset related
    parser.add_argument('--train_dir', type=str, default='train2017', help='train data dir')
    parser.add_argument('--train_ann', type=str, default='person_keypoints_train2017.json',
                        help='train annotations json')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')

    args, _ = parser.parse_known_args()

    args.jsonpath_train = os.path.join(params['data_dir'], 'annotations/' + args.train_ann)
    args.imgpath_train = os.path.join(params['data_dir'], args.train_dir)
    args.maskpath_train = os.path.join(params['data_dir'], 'ignore_mask_train')

    return args


def get_lr(lr, lr_gamma, steps_per_epoch, max_epoch_train, lr_steps, group_size):
    lr_stage = np.array([lr] * steps_per_epoch * max_epoch_train).astype('f')
    for step in lr_steps:
        step //= group_size
        lr_stage[step:] *= lr_gamma

    lr_base = lr_stage.copy()
    lr_base = lr_base / 4

    lr_vgg = lr_base.copy()
    vgg_freeze_step = 2000
    lr_vgg[:vgg_freeze_step] = 0
    return lr_stage, lr_base, lr_vgg

# zhang add
def adjust_learning_rate(init_lr, lr_gamma, steps_per_epoch, max_epoch_train, stepvalues):
    lr_stage = np.array([init_lr] * steps_per_epoch * max_epoch_train).astype('f')
    for epoch in stepvalues:
        lr_stage[epoch * steps_per_epoch:] *= lr_gamma

    lr_base = lr_stage.copy()
    lr_base = lr_base / 4

    lr_vgg = lr_base.copy()
    vgg_freeze_step = 2000
    lr_vgg[:vgg_freeze_step] = 0
    return lr_stage, lr_base, lr_vgg


def load_model(test_net, model_path):
    if model_path:
        param_dict = load_checkpoint(model_path)
        # print(type(param_dict))
        param_dict_new = {}
        for key, values in param_dict.items():
            # print('key:', key)
            if key.startswith('moment'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values

            # else:
            # param_dict_new[key] = values
        load_param_into_net(test_net, param_dict_new)


class show_loss_list():
    def __init__(self, name):
        self.loss_list = np.zeros(6).astype('f')
        self.sums = 0
        self.name = name

    def add(self, list_of_tensor):
        self.sums += 1
        for i, loss_tensor in enumerate(list_of_tensor):
            self.loss_list[i] += loss_tensor.asnumpy()

    def show(self):
        print(self.name + ' stage_loss:', self.loss_list / (self.sums + 1e-8), flush=True)
        self.loss_list = np.zeros(6).astype('f')
        self.sums = 0


class AverageMeter():
    def __init__(self):
        self.loss = 0
        self.sum = 0

    def add(self, tensor):
        self.sum += 1
        self.loss += tensor.asnumpy()

    def meter(self):
        avergeLoss = self.loss / (self.sum + 1e-8)
        self.loss = 0
        self.sum = 0
        return avergeLoss
