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

import time
import numpy as np
from PIL import Image
from mindspore import nn
from mindspore.ops import operations as ops
from mindspore.train.callback import Callback
from mindspore.common.tensor import Tensor

class UnetEval(nn.Cell):
    """
    Add Unet evaluation activation.
    """
    def __init__(self, net):
        super(UnetEval, self).__init__()
        self.net = net
        self.transpose = ops.Transpose()
        self.softmax = ops.Softmax(axis=-1)
        self.argmax = ops.Argmax(axis=-1)

    def construct(self, x):
        out = self.net(x)
        out = self.transpose(out, (0, 2, 3, 1))
        softmax_out = self.softmax(out)
        argmax_out = self.argmax(out)
        return (softmax_out, argmax_out)

class StepLossTimeMonitor(Callback):

    def __init__(self, batch_size, per_print_times=1):
        super(StepLossTimeMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.batch_size = batch_size

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):

        step_seconds = time.time() - self.step_time
        step_fps = self.batch_size*1.0/step_seconds

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
        self.losses.append(loss)
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            # TEST
            print("step: %s, loss is %s, fps is %s" % (cur_step_in_epoch, loss, step_fps), flush=True)

    def epoch_begin(self, run_context):
        self.epoch_start = time.time()
        self.losses = []

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_cost = time.time() - self.epoch_start
        step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        step_fps = self.batch_size * 1.0 * step_in_epoch / epoch_cost
        print("epoch: {:3d}, avg loss:{:.4f}, total cost: {:.3f} s, per step fps:{:5.3f}".format(
            cb_params.cur_epoch_num, np.mean(self.losses), epoch_cost, step_fps), flush=True)

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def filter_checkpoint_parameter_by_list(param_dict, filter_list):
    """remove useless parameters according to filter_list"""
    for key in list(param_dict.keys()):
        for name in filter_list:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del param_dict[key]
                break
