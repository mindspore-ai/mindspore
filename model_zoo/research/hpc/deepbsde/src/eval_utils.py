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

import time
from mindspore import Tensor, save_checkpoint
from mindspore.train.callback import Callback

class EvalCallBack(Callback):
    """
    Evaluation callback when training.

    Args:
        eval_param_dict (dict): evaluation parameters' configure dict.
        ckpt_path (str): save checkpoint path format, eg: "./logs/deepbsde_hjb_{}.ckpt".
        interval (int): run evaluation interval, default is 1.

    Returns:
        None

    Examples:
        >>> EvalCallBack(eval_function, eval_param_dict)
    """

    def __init__(self, eval_param_dict, ckpt_path, interval=1):
        super(EvalCallBack, self).__init__()
        self.eval_param_dict = eval_param_dict
        self.eval_function = apply_eval
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.best_res = 0
        self.best_epoch = 0
        self.ckpt_path = ckpt_path
        self.start_time = time.time()

    def epoch_end(self, run_context):
        """Callback when epoch end."""
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if cur_step % self.interval == 0:
            loss, y_init = self.eval_function(self.eval_param_dict)
            elapsed_time = time.time() - self.start_time
            print("total step: {:4d}, eval loss: {:5.3f}, Y0: {:5.3f}, elapsed time: {:3.0f}".format(
                cur_step, loss, y_init, elapsed_time))

    def end(self, run_context):
        cb_params = run_context.original_args()
        save_checkpoint(cb_params.train_network, self.ckpt_path.format("end"))

def apply_eval(eval_param):
    eval_model = eval_param["model"]
    dw, x = eval_param["valid_data"]
    eval_model.set_train(False)
    loss = eval_model(Tensor(dw), Tensor(x)).asnumpy()
    y_init = eval_model.net.y_init.asnumpy()[0]
    return loss, y_init
