# Copyright 2025 Huawei Technologies Co., Ltd
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
""" tre train"""

import mindspore as ms
from mindspore import nn
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, Callback, TrainFaultTolerance
import mindspore.dataset as ds
import mindspore.runtime as rt
from mindspore.common.initializer import Normal
from mindspore.ops import operations as P
import numpy as np
import os
import sys

rt.set_memory(max_size="2GB")
ms.set_seed(11)
np.random.seed(11)

class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Number of classes. Default: 10.
        num_channel (int): Number of channels. Default: 1.
        include_top (bool): If includes fc layers. Default: True.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> LeNet5(num_class=10)

    """
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))


    def construct(self, x):
        '''
        Forward network.
        '''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (-1, 400))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_dataset(batch_size):
    """create dataset"""
    num_elems = batch_size * 20
    data = np.random.randn(num_elems, 1, 32, 32).astype(np.float32)
    label = np.random.randint(low=0, high=10, size=num_elems, dtype=np.int32)
    return ds.NumpySlicesDataset({"data": data, "label": label}, shuffle=False).batch(batch_size)

# flag for marking whether TREError occurred
training_error_occured = False
# recording loss before error occurred
first_time_loss = {}
# recording loss before error occurred
resume_train_loss = {}

def check_loss_consistency():
    """check the loss after resume training is same as before"""
    num_retraining_steps = 0
    for step, loss_after in resume_train_loss.items():
        if not step in first_time_loss:
            continue
        num_retraining_steps += 1
        loss_before = first_time_loss[step]
        if not np.isclose(loss_before, loss_after):
            print(f'ERROR: loss values of step {step} before({loss_before}) '
                  f'and after({loss_after}) resume training are not same')
            return False
    if num_retraining_steps <= 0:
        print('ERROR: there are no overlapped steps before and after resume training')
        return False
    return True


class MyLossMonitor(Callback):
    """
    Self defined loss monitor
    """
    def __init__(self, per_print_times=1):
        super().__init__()
        self._per_print_times = per_print_times
        self._last_print_time = 0

    def on_train_step_end(self, run_context):
        """
        Print training loss at the end of step.
        """
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        loss = float(np.mean(cb_params.net_outputs.asnumpy()))

        if isinstance(cb_params.initial_step, int):
            cur_step_in_epoch = (cb_params.initial_step + cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        else:
            cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("In epoch: {} step: {}, loss is NAN or INF, training process cannot continue, "
                             "terminating training.".format(cur_epoch_num, cur_step_in_epoch))

        # In disaster recovery scenario, the cb_params.cur_step_num may be rollback to previous step
        # and be less than self._last_print_time, so self._last_print_time need to be updated.
        if self._per_print_times != 0 and (cb_params.cur_step_num <= self._last_print_time):
            while cb_params.cur_step_num <= self._last_print_time:
                self._last_print_time -=\
                    max(self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            print("epoch: %s step: %s, loss is %s" % (cur_epoch_num, cur_step_in_epoch, loss), flush=True)

        # record training loss
        global training_error_occured
        if training_error_occured:
            resume_train_loss[cur_step_in_epoch] = loss
        else:
            first_time_loss[cur_step_in_epoch] = loss

        loss_low_thresh = 8000.0
        loss_high_thresh = 8500.0
        if loss_low_thresh < loss < loss_high_thresh  and not training_error_occured:
            training_error_occured = True
            raise RuntimeError(f"TREError occurred, current loss {loss}, "
                               f"loss thresh values are ({loss_low_thresh}, {loss_high_thresh})")


if __name__ == '__main__':
    ckpt_path = "./checkpoints"
    os.system(f'rm -rf {ckpt_path}')
    ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    dataset = create_dataset(batch_size=32)
    net = LeNet5()
    loss_func = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    loss_scale_manager = ms.FixedLossScaleManager(1024., False)
    optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(net, loss_fn=loss_func, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=10, keep_checkpoint_max=2)
    ckpt_cb = ModelCheckpoint(prefix="simple_net", directory=ckpt_path, config=ckpt_cfg)
    loss_cb = MyLossMonitor()

    def ckpt_load_func():
        print('Begin to load checkpoint')
        ckpt_file = f"{ckpt_path}/simple_net-1_10.ckpt"
        param_dict = ms.load_checkpoint(ckpt_file)
        print(f'End to load ckpt, param_dict.size={len(param_dict)}')
        # set resume traning epoch and step
        param_dict['epoch_num'] = ms.Parameter(ms.Tensor(1, ms.int32), name='epoch_num')
        param_dict['step_num'] = ms.Parameter(ms.Tensor(10, ms.int32), name='step_num')
        return param_dict, False

    default_args = {
        "ckpt_load_fn": ckpt_load_func,
    }
    tft_cb = TrainFaultTolerance(ckpt_save_path=ckpt_path, ckpt_load_fn=ckpt_load_func)

    model.train(1, dataset, callbacks=[ckpt_cb, loss_cb, tft_cb])
    sys.exit(0 if check_loss_consistency() else 1)
