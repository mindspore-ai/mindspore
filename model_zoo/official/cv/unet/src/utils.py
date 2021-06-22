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
import os
import time
import shutil
import cv2
import numpy as np
from PIL import Image
from mindspore import nn
from mindspore.ops import operations as ops
from mindspore.train.callback import Callback
from mindspore.common.tensor import Tensor
from src.model_utils.config import config


class UnetEval(nn.Cell):
    """
    Add Unet evaluation activation.
    """

    def __init__(self, net, need_slice=False, eval_activate="softmax"):
        super(UnetEval, self).__init__()
        self.net = net
        self.need_slice = need_slice
        self.transpose = ops.Transpose()
        self.softmax = ops.Softmax(axis=-1)
        self.argmax = ops.Argmax(axis=-1)
        self.squeeze = ops.Squeeze(axis=0)
        if eval_activate.lower() not in ("softmax", "argmax"):
            raise ValueError("eval_activate only support 'softmax' or 'argmax'")
        self.is_softmax = True
        if eval_activate == "argmax":
            self.is_softmax = False

    def construct(self, x):
        out = self.net(x)
        if self.need_slice:
            out = self.squeeze(out[-1:])
        out = self.transpose(out, (0, 2, 3, 1))
        if self.is_softmax:
            softmax_out = self.softmax(out)
            return softmax_out
        argmax_out = self.argmax(out)
        return argmax_out


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

    def __init__(self, print_res=True, show_eval=False):
        super(dice_coeff, self).__init__()
        self.clear()
        self.show_eval = show_eval
        self.print_res = print_res
        self.img_num = 0

    def clear(self):
        self._dice_coeff_sum = 0
        self._iou_sum = 0
        self._samples_num = 0
        self.img_num = 0
        self.eval_images_path = "./draw_eval"
        if os.path.exists(self.eval_images_path):
            shutil.rmtree(self.eval_images_path)
        os.mkdir(self.eval_images_path)

    def draw_img(self, gray, index):
        """
        black：rgb(0,0,0)
        red：rgb(255,0,0)
        green：rgb(0,255,0)
        blue：rgb(0,0,255)
        cyan：rgb(0,255,255)
        cyan purple：rgb(255,0,255)
        white：rgb(255,255,255)
        """
        color = config.color
        color = np.array(color)
        np_draw = np.uint8(color[gray.astype(int)])
        return np_draw

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Need 2 inputs (y_predict, y), but got {}'.format(len(inputs)))
        y = self._convert_data(inputs[1])
        self._samples_num += y.shape[0]
        y = y.transpose(0, 2, 3, 1)
        b, h, w, c = y.shape
        if b != 1:
            raise ValueError('Batch size should be 1 when in evaluation.')
        y = y.reshape((h, w, c))
        start_index = 0
        if not config.include_background:
            y = y[:, :, 1:]
            start_index = 1

        if config.eval_activate.lower() == "softmax":
            y_softmax = np.squeeze(self._convert_data(inputs[0]), axis=0)
            if config.eval_resize:
                y_pred = []
                for i in range(start_index, config.num_classes):
                    y_pred.append(cv2.resize(np.uint8(y_softmax[:, :, i] * 255), (w, h)) / 255)
                y_pred = np.stack(y_pred, axis=-1)
            else:
                y_pred = y_softmax
                if not config.include_background:
                    y_pred = y_softmax[:, :, start_index:]

        elif config.eval_activate.lower() == "argmax":
            y_argmax = np.squeeze(self._convert_data(inputs[0]), axis=0)
            y_pred = []
            for i in range(start_index, config.num_classes):
                if config.eval_resize:
                    y_pred.append(cv2.resize(np.uint8(y_argmax == i), (w, h), interpolation=cv2.INTER_NEAREST))
                else:
                    y_pred.append(np.float32(y_argmax == i))
            y_pred = np.stack(y_pred, axis=-1)
        else:
            raise ValueError('config eval_activate should be softmax or argmax.')

        if self.show_eval:
            self.img_num += 1
            if not config.include_background:
                y_pred_draw = np.ones((h, w, c)) * 0.5
                y_pred_draw[:, :, 1:] = y_pred
                y_draw = np.ones((h, w, c)) * 0.5
                y_draw[:, :, 1:] = y
            else:
                y_pred_draw = y_pred
                y_draw = y
            y_pred_draw = y_pred_draw.argmax(-1)
            y_draw = y_draw.argmax(-1)
            cv2.imwrite(os.path.join(self.eval_images_path, "predict-" + str(self.img_num) + ".png"),
                        self.draw_img(y_pred_draw, 2))
            cv2.imwrite(os.path.join(self.eval_images_path, "mask-" + str(self.img_num) + ".png"),
                        self.draw_img(y_draw, 2))

        y_pred = y_pred.astype(np.float32)
        inter = np.dot(y_pred.flatten(), y.flatten())
        union = np.dot(y_pred.flatten(), y_pred.flatten()) + np.dot(y.flatten(), y.flatten())

        single_dice_coeff = 2 * float(inter) / float(union + 1e-6)
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
        step_fps = self.batch_size * 1.0 / step_seconds

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
