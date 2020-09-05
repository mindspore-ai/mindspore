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

"""tinybert utils"""

import os
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.train.callback import Callback
from mindspore.train.serialization import save_checkpoint
from mindspore.ops import operations as P
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR
from .assessment_method import Accuracy

class ModelSaveCkpt(Callback):
    """
    Saves checkpoint.
    If the loss in NAN or INF terminating training.
    Args:
        network (Network): The train network for training.
        save_ckpt_num (int): The number to save checkpoint, default is 1000.
        max_ckpt_num (int): The max checkpoint number, default is 3.
    """
    def __init__(self, network, save_ckpt_step, max_ckpt_num, output_dir):
        super(ModelSaveCkpt, self).__init__()
        self.count = 0
        self.network = network
        self.save_ckpt_step = save_ckpt_step
        self.max_ckpt_num = max_ckpt_num
        self.output_dir = output_dir

    def step_end(self, run_context):
        """step end and save ckpt"""
        cb_params = run_context.original_args()
        if cb_params.cur_step_num % self.save_ckpt_step == 0:
            saved_ckpt_num = cb_params.cur_step_num / self.save_ckpt_step
            if saved_ckpt_num > self.max_ckpt_num:
                oldest_ckpt_index = saved_ckpt_num - self.max_ckpt_num
                path = os.path.join(self.output_dir, "tiny_bert_{}_{}.ckpt".format(int(oldest_ckpt_index),
                                                                                   self.save_ckpt_step))
                if os.path.exists(path):
                    os.remove(path)
            save_checkpoint(self.network, os.path.join(self.output_dir,
                                                       "tiny_bert_{}_{}.ckpt".format(int(saved_ckpt_num),
                                                                                     self.save_ckpt_step)))

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0")
        self._per_print_times = per_print_times

    def step_end(self, run_context):
        """step end and print loss"""
        cb_params = run_context.original_args()
        print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num,
                                                           cb_params.cur_step_num,
                                                           str(cb_params.net_outputs)))

class EvalCallBack(Callback):
    """Evaluation callback"""
    def __init__(self, network, dataset):
        super(EvalCallBack, self).__init__()
        self.network = network
        self.global_acc = 0.0
        self.dataset = dataset

    def step_end(self, run_context):
        """step end and do evaluation"""
        cb_params = run_context.original_args()
        if cb_params.cur_step_num % 100 == 0:
            callback = Accuracy()
            columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
            for data in self.dataset.create_dict_iterator():
                input_data = []
                for i in columns_list:
                    input_data.append(data[i])
                input_ids, input_mask, token_type_id, label_ids = input_data
                self.network.set_train(False)
                logits = self.network(input_ids, token_type_id, input_mask)
                callback.update(logits[3], label_ids)
            acc = callback.acc_num / callback.total_num
            with open("./eval.log", "a+") as f:
                f.write("acc_num {}, total_num{}, accuracy{:.6f}".format(callback.acc_num, callback.total_num,
                                                                         callback.acc_num / callback.total_num))
                f.write('\n')

            if acc > self.global_acc:
                self.global_acc = acc
                print("The best acc is {}".format(acc))
                eval_model_ckpt_file = "eval_model.ckpt"
                if os.path.exists(eval_model_ckpt_file):
                    os.remove(eval_model_ckpt_file)
                save_checkpoint(self.network, eval_model_ckpt_file)

class BertLearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for Bert network.
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(BertLearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr
