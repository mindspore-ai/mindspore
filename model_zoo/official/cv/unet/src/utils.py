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
import cv2
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
    def __init__(self, net, need_slice=False):
        super(UnetEval, self).__init__()
        self.net = net
        self.need_slice = need_slice
        self.transpose = ops.Transpose()
        self.softmax = ops.Softmax(axis=-1)
        self.argmax = ops.Argmax(axis=-1)
        self.squeeze = ops.Squeeze(axis=0)

    def construct(self, x):
        out = self.net(x)
        if self.need_slice:
            out = self.squeeze(out[-1:])
        out = self.transpose(out, (0, 2, 3, 1))
        softmax_out = self.softmax(out)
        argmax_out = self.argmax(out)
        return (softmax_out, argmax_out)

class TempLoss(nn.Cell):
    """A temp loss cell."""
    def __init__(self):
        super(TempLoss, self).__init__()
        self.identity = ops.identity()
    def construct(self, logits, label):
        return self.identity(logits)

def apply_eval(eval_param_dict):
    """run Evaluation"""
    model = eval_param_dict["model"]
    dataset = eval_param_dict["dataset"]
    metrics_name = eval_param_dict["metrics_name"]
    index = 0 if metrics_name == "dice_coeff" else 1
    eval_score = model.eval(dataset, dataset_sink_mode=False)["dice_coeff"][index]
    return eval_score

class dice_coeff(nn.Metric):
    """Unet Metric, return dice coefficient and IOU."""
    def __init__(self, cfg_unet, print_res=True):
        super(dice_coeff, self).__init__()
        self.clear()
        self.cfg_unet = cfg_unet
        self.print_res = print_res

    def clear(self):
        self._dice_coeff_sum = 0
        self._iou_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Need 2 inputs ((y_softmax, y_argmax), y), but got {}'.format(len(inputs)))
        y = self._convert_data(inputs[1])
        self._samples_num += y.shape[0]
        y = y.transpose(0, 2, 3, 1)
        b, h, w, c = y.shape
        if b != 1:
            raise ValueError('Batch size should be 1 when in evaluation.')
        y = y.reshape((h, w, c))
        if self.cfg_unet["eval_activate"].lower() == "softmax":
            y_softmax = np.squeeze(self._convert_data(inputs[0][0]), axis=0)
            if self.cfg_unet["eval_resize"]:
                y_pred = []
                for i in range(self.cfg_unet["num_classes"]):
                    y_pred.append(cv2.resize(np.uint8(y_softmax[:, :, i] * 255), (w, h)) / 255)
                y_pred = np.stack(y_pred, axis=-1)
            else:
                y_pred = y_softmax
        elif self.cfg_unet["eval_activate"].lower() == "argmax":
            y_argmax = np.squeeze(self._convert_data(inputs[0][1]), axis=0)
            y_pred = []
            for i in range(self.cfg_unet["num_classes"]):
                if self.cfg_unet["eval_resize"]:
                    y_pred.append(cv2.resize(np.uint8(y_argmax == i), (w, h), interpolation=cv2.INTER_NEAREST))
                else:
                    y_pred.append(np.float32(y_argmax == i))
            y_pred = np.stack(y_pred, axis=-1)
        else:
            raise ValueError('config eval_activate should be softmax or argmax.')
        y_pred = y_pred.astype(np.float32)
        inter = np.dot(y_pred.flatten(), y.flatten())
        union = np.dot(y_pred.flatten(), y_pred.flatten()) + np.dot(y.flatten(), y.flatten())

        single_dice_coeff = 2 * float(inter) / float(union+1e-6)
        single_iou = single_dice_coeff / (2 - single_dice_coeff)
        if self.print_res:
            print("single dice coeff is: {}, IOU is: {}".format(single_dice_coeff, single_iou))
        self._dice_coeff_sum += single_dice_coeff
        self._iou_sum += single_iou

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return (self._dice_coeff_sum / float(self._samples_num), self._iou_sum / float(self._samples_num))

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
