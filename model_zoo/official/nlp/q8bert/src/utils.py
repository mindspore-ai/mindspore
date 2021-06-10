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
# ===========================================================================

"""q8bert utils"""

import os
import logging
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.train.callback import Callback
from mindspore.train.serialization import save_checkpoint
from mindspore.ops import operations as P
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR
from .config import glue_output_modes

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def glue_compute_metrics(task_name, preds, labels):
        """different dataset evaluation."""
        assert len(preds) == len(labels)
        if task_name == "cola":
            result = {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            result = {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            result = acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            result = pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            result = acc_and_f1(preds, labels)
        elif task_name == "mnli":
            result = {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            result = {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            result = {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            result = {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            result = {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
        return result


prior_index = {
    "cola": "mcc",
    "mnli": "acc",
    "mnli-mm": "acc",
    "mrpc": "acc",
    "sst-2": "acc",
    "sts-b": "pearson",
    "qqp": "acc",
    "qnli": "acc",
    "rte": "acc",
    "wnli": "acc",
}


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
                path = os.path.join(self.output_dir, "q8bert_{}_{}.ckpt".format(int(oldest_ckpt_index),
                                                                                cb_params.cur_step_num))
                if os.path.exists(path):
                    os.remove(path)
            save_checkpoint(self.network, os.path.join(self.output_dir,
                                                       "q8bert_{}_{}.ckpt".format(int(saved_ckpt_num),
                                                                                  cb_params.cur_step_num)))


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
    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0")
        self._per_print_times = per_print_times

    def step_end(self, run_context):
        """step end and print loss"""
        cb_params = run_context.original_args()
        loss, _ = cb_params.net_outputs
        print("epoch: {}, step: {}, loss: {}".format(cb_params.cur_epoch_num,
                                                     cb_params.cur_step_num,
                                                     loss))


class EvalCallBack(Callback):
    """Evaluation callback"""
    def __init__(self, network, dataset, task_name, logging_step, save_ckpt_dir):
        super(EvalCallBack, self).__init__()
        self.network = network
        self.global_acc = 0.0
        self.dataset = dataset
        self.task_name = task_name
        self.logging_step = logging_step
        self.best_result = 0.0
        self.save_ckpt_dir = save_ckpt_dir

    def step_end(self, run_context):
        """step end and do evaluation"""
        cb_params = run_context.original_args()
        label_nums = 2
        if self.task_name.lower == 'mnli':
            label_nums = 3
        if cb_params.cur_step_num % self.logging_step == 0:
            self.network.set_train(False)
            columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
            preds = None
            out_label_ids = None
            for data in self.dataset.create_dict_iterator(num_epochs=1):
                input_data = []
                for i in columns_list:
                    input_data.append(data[i])
                input_ids, input_mask, token_type_id, label_ids = input_data
                _, _, logits, _ = self.network(input_ids, token_type_id, input_mask)
                if preds is None:
                    preds = logits.asnumpy()
                    preds = np.reshape(preds, [-1, label_nums])
                    out_label_ids = label_ids.asnumpy()
                else:
                    preds = np.concatenate((preds, np.reshape(logits.asnumpy(), [-1, label_nums])), axis=0)
                    out_label_ids = np.append(out_label_ids, label_ids.asnumpy())
            if glue_output_modes[self.task_name.lower()] == "classification":
                preds = np.argmax(preds, axis=1)
            elif glue_output_modes[self.task_name.lower()] == "regression":
                preds = np.reshape(preds, [-1])
            result = glue_compute_metrics(self.task_name.lower(), preds, out_label_ids)
            prior_result = result[prior_index[self.task_name.lower()]]
            if prior_result > self.best_result:
                self.best_result = prior_result
                eval_model_ckpt_file = os.path.join(self.save_ckpt_dir, self.task_name.lower() + "_eval_model.ckpt")
                if os.path.exists(eval_model_ckpt_file):
                    os.remove(eval_model_ckpt_file)
                save_checkpoint(self.network, eval_model_ckpt_file)
            print("The current result is {}, the best result is {}".format(result, self.best_result))
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
