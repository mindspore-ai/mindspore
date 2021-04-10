import math
import time
import numpy as np

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import LossMonitor, Callback
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype

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
            self._loss_list.append(loss)
            if cb_params.cur_step_num % 100 == 0:
                print("epoch: %s, steps: [%s], mean loss is: %s"%(cb_params.cur_epoch_num, cur_step_in_epoch,
                                                                  np.array(self._loss_list).mean()), flush=True)
                self._loss_list = []

        self._start_time = time.time()

class MyScaleSensCallback(Callback):
    '''MyLossScaleCallback'''
    def __init__(self, loss_scale_list, epoch_list):
        super(MyScaleSensCallback, self).__init__()
        self.loss_scale_list = loss_scale_list
        self.epoch_list = epoch_list
        self.scaling_sens = loss_scale_list[0]

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num

        for i, _ in enumerate(self.epoch_list):
            if epoch >= self.epoch_list[i]:
                self.scaling_sens = self.loss_scale_list[i+1]
            else:
                break

        scaling_sens_tensor = Tensor(self.scaling_sens, dtype=mstype.float32)
        cb_params.train_network.set_sense_scale(scaling_sens_tensor)
        print("Epoch: set train network scale sens to {}".format(self.scaling_sens))


def _linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate

def _a_cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
    base = float(current_step - warmup_steps) / float(decay_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate

def _dynamic_lr(base_lr, total_steps, warmup_steps, warmup_ratio=1 / 3):
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr.append(_linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * warmup_ratio))
        else:
            lr.append(_a_cosine_learning_rate(i, base_lr, warmup_steps, total_steps))

    return lr

def get_lr(lr, lr_gamma, steps_per_epoch, max_epoch_train, lr_steps, group_size, lr_type='default', warmup_epoch=5):
    if lr_type == 'default':
        lr_stage = np.array([lr] * steps_per_epoch * max_epoch_train).astype('f')
        for step in lr_steps:
            step //= group_size
            lr_stage[step:] *= lr_gamma
    elif lr_type == 'cosine':
        lr_stage = _dynamic_lr(lr, steps_per_epoch * max_epoch_train, warmup_epoch * steps_per_epoch,
                               warmup_ratio=1 / 3)
        lr_stage = np.array(lr_stage).astype('f')
    else:
        raise ValueError("lr type {} is not support.".format(lr_type))

    lr_base = lr_stage.copy()
    lr_base = lr_base / 4
    lr_vgg = lr_base.copy()
    vgg_freeze_step = 2000 // group_size
    lr_vgg[:vgg_freeze_step] = 0

    return lr_stage, lr_base, lr_vgg


def load_model(test_net, model_path):
    if model_path:
        param_dict = load_checkpoint(model_path)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moment'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values

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
