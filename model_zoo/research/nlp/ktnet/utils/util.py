# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Functional Cells used in finetune and evaluation."""


import os
import math
from mindspore import log as logger
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.train.callback import Callback
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore._checkparam import Validator as validator


def make_directory(path: str):
    """Make directory."""
    if path is None or not isinstance(path, str) or path.strip() == "":
        logger.error("The path(%r) is invalid type.", path)
        raise TypeError("Input path is invalid type")

    # convert the relative paths
    path = os.path.realpath(path)
    logger.debug("The abs path is %r", path)

    # check the path is exist and write permissions?
    if os.path.exists(path):
        real_path = path
    else:
        # All exceptions need to be caught because create directory maybe have some limit(permissions)
        logger.debug("The directory(%s) doesn't exist, will create it", path)
        try:
            os.makedirs(path, exist_ok=True)
            real_path = path
        except PermissionError as e:
            logger.error("No write permission on the directory(%r), error = %r", path, e)
            raise TypeError("No write permission on the directory.")
    return real_path


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, dataset_size=-1):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size

    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num / self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            print("epoch: {}, current epoch percent: {}, step: {}, outputs are {}"
                  .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, str(cb_params.net_outputs)),
                  flush=True)
        else:
            print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                               str(cb_params.net_outputs)), flush=True)


def LoadNewestCkpt(load_finetune_checkpoint_dir, steps_per_epoch, epoch_num, prefix):
    """
    Find the ckpt finetune generated and load it into eval network.
    """
    files = os.listdir(load_finetune_checkpoint_dir)
    pre_len = len(prefix)
    max_num = 0
    for filename in files:
        name_ext = os.path.splitext(filename)
        if name_ext[-1] != ".ckpt":
            continue
        if filename.find(prefix) == 0 and not filename[pre_len].isalpha():
            index = filename[pre_len:].find("-")
            if index == 0 and max_num == 0:
                load_finetune_checkpoint_path = os.path.join(load_finetune_checkpoint_dir, filename)
            elif index not in (0, -1):
                name_split = name_ext[-2].split('_')
                if (steps_per_epoch != int(name_split[len(name_split) - 1])) \
                        or (epoch_num != int(filename[pre_len + index + 1:pre_len + index + 2])):
                    continue
                num = filename[pre_len + 1:pre_len + index]
                if int(num) > max_num:
                    max_num = int(num)
                    load_finetune_checkpoint_path = os.path.join(load_finetune_checkpoint_dir, filename)
    return load_finetune_checkpoint_path


class CustomWarmUpLR(LearningRateSchedule):
    """
    apply the functions to  the corresponding input fields.
    ·
    """

    def __init__(self, learning_rate, warmup_steps, max_train_steps):
        super(CustomWarmUpLR, self).__init__()
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be float.")
        validator.check_non_negative_float(learning_rate, "learning_rate", self.cls_name)
        validator.check_positive_int(warmup_steps, 'warmup_steps', self.cls_name)
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.max_train_steps = max_train_steps
        self.cast = P.Cast()

    def construct(self, current_step):
        if current_step < self.warmup_steps:
            warmup_percent = self.cast(current_step, mstype.float32) / self.warmup_steps
        else:
            warmup_percent = 1 - self.cast(current_step, mstype.float32) / self.max_train_steps

        return self.learning_rate * warmup_percent
