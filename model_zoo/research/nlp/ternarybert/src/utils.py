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

"""ternarybert utils"""

import os
import time
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.train.callback import Callback
from mindspore.train.serialization import save_checkpoint
from mindspore.ops import operations as P
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR
from .quant import convert_network, save_params, restore_params


class ModelSaveCkpt(Callback):
    """
    Saves checkpoint.
    If the loss in NAN or INF terminating training.
    Args:
        network (Network): The train network for training.
        save_ckpt_step (int): The step to save checkpoint.
        max_ckpt_num (int): The max checkpoint number.
    """
    def __init__(self, network, save_ckpt_step, max_ckpt_num, output_dir, embedding_bits=2, weight_bits=2,
                 clip_value=1.0):
        super(ModelSaveCkpt, self).__init__()
        self.count = 0
        self.network = network
        self.save_ckpt_step = save_ckpt_step
        self.max_ckpt_num = max_ckpt_num
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.embedding_bits = embedding_bits
        self.weight_bits = weight_bits
        self.clip_value = clip_value

    def step_end(self, run_context):
        """step end and save ckpt"""
        cb_params = run_context.original_args()
        if cb_params.cur_step_num % self.save_ckpt_step == 0:
            saved_ckpt_num = cb_params.cur_step_num / self.save_ckpt_step
            if saved_ckpt_num > self.max_ckpt_num:
                oldest_ckpt_index = saved_ckpt_num - self.max_ckpt_num
                path = os.path.join(self.output_dir, "ternary_bert_{}_{}.ckpt".format(int(oldest_ckpt_index),
                                                                                      self.save_ckpt_step))
                if os.path.exists(path):
                    os.remove(path)
            params_dict = save_params(self.network)
            convert_network(self.network, self.embedding_bits, self.weight_bits, self.clip_value)
            save_checkpoint(self.network, os.path.join(self.output_dir,
                                                       "ternary_bert_{}_{}.ckpt".format(int(saved_ckpt_num),
                                                                                        self.save_ckpt_step)))
            restore_params(self.network, params_dict)


class LossCallBack(Callback):
    """
    Monitor the loss in training.
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


class StepCallBack(Callback):
    """
    Monitor the time in training.
    """
    def __init__(self):
        super(StepCallBack, self).__init__()
        self.start_time = 0.0

    def step_begin(self, run_context):
        self.start_time = time.time()

    def step_end(self, run_context):
        time_cost = time.time() - self.start_time
        cb_params = run_context.original_args()
        print("step: {}, second_per_step: {}".format(cb_params.cur_step_num, time_cost))


class EvalCallBack(Callback):
    """Evaluation callback"""
    def __init__(self, network, dataset, eval_ckpt_step, save_ckpt_dir, embedding_bits=2, weight_bits=2,
                 clip_value=1.0, metrics=None):
        super(EvalCallBack, self).__init__()
        self.network = network
        self.global_metrics = 0.0
        self.dataset = dataset
        self.eval_ckpt_step = eval_ckpt_step
        self.save_ckpt_dir = save_ckpt_dir
        self.embedding_bits = embedding_bits
        self.weight_bits = weight_bits
        self.clip_value = clip_value
        self.metrics = metrics
        if not os.path.exists(save_ckpt_dir):
            os.makedirs(save_ckpt_dir)

    def step_end(self, run_context):
        """step end and do evaluation"""
        cb_params = run_context.original_args()
        if cb_params.cur_step_num % self.eval_ckpt_step == 0:
            params_dict = save_params(self.network)
            convert_network(self.network, self.embedding_bits, self.weight_bits, self.clip_value)
            self.network.set_train(False)
            callback = self.metrics()
            columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
            for data in self.dataset:
                input_data = []
                for i in columns_list:
                    input_data.append(data[i])
                input_ids, input_mask, token_type_id, label_ids = input_data
                _, _, logits, _ = self.network(input_ids, token_type_id, input_mask)
                callback.update(logits, label_ids)
            metrics = callback.get_metrics()

            if metrics > self.global_metrics:
                self.global_metrics = metrics
                eval_model_ckpt_file = os.path.join(self.save_ckpt_dir, 'eval_model.ckpt')
                if os.path.exists(eval_model_ckpt_file):
                    os.remove(eval_model_ckpt_file)
                save_checkpoint(self.network, eval_model_ckpt_file)
            print('step {}, {} {}, best_{} {}'.format(cb_params.cur_step_num,
                                                      callback.name,
                                                      metrics,
                                                      callback.name,
                                                      self.global_metrics))
            restore_params(self.network, params_dict)
            self.network.set_train(True)


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
