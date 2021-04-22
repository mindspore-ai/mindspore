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
"""Evaluation callback when training"""

import os
import stat
import time
from mindspore import save_checkpoint
from mindspore import log as logger
from mindspore.train.callback import Callback


class EvalCallBack(Callback):
    """
    Evaluation callback when training.

    Args:
        eval_function (function): evaluation function.
        eval_param_dict (dict): evaluation parameters' configure dict.
        interval (int): run evaluation interval, default is 1.
        eval_start_epoch (int): evaluation start epoch, default is 1.
        save_best_ckpt (bool): Whether to save best checkpoint, default is True.
        besk_ckpt_name (str): bast checkpoint name, default is `best.ckpt`.
        metrics_name (str): evaluation metrics name, default is `acc`.

    Returns:
        None

    Examples:
        >>> EvalCallBack(eval_function, eval_param_dict)
    """

    def __init__(self, eval_function, eval_param_dict, interval=1, eval_start_epoch=1, save_best_ckpt=True,
                 ckpt_directory="./", besk_ckpt_name="best.ckpt", metrics_name="acc"):
        super(EvalCallBack, self).__init__()
        self.eval_param_dict = eval_param_dict
        self.eval_function = eval_function
        self.eval_start_epoch = eval_start_epoch
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.best_res = 0
        self.best_epoch = 0
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.bast_ckpt_path = os.path.join(ckpt_directory, besk_ckpt_name)
        self.metrics_name = metrics_name

    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
        except OSError:
            logger.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            logger.warning("ValueError, failed to remove the older ckpt file %s.", file_name)

    def epoch_end(self, run_context):
        """Callback when epoch end."""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            eval_start = time.time()
            res = self.eval_function(self.eval_param_dict)
            eval_cost = time.time() - eval_start
            print("epoch: {}, {}: {}, eval_cost:{:.2f}".format(cur_epoch, self.metrics_name, res, eval_cost),
                  flush=True)
            if res >= self.best_res:
                self.best_res = res
                self.best_epoch = cur_epoch
                print("update best result: {}".format(res), flush=True)
                if self.save_best_ckpt:
                    if os.path.exists(self.bast_ckpt_path):
                        self.remove_ckpoint_file(self.bast_ckpt_path)
                    save_checkpoint(cb_params.train_network, self.bast_ckpt_path)
                    print("update best checkpoint at: {}".format(self.bast_ckpt_path), flush=True)

    def end(self, run_context):
        print("End training, the best {0} is: {1}, the best {0} epoch is {2}".format(self.metrics_name,
                                                                                     self.best_res,
                                                                                     self.best_epoch), flush=True)
