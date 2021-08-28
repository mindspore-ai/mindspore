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

"""
Functional Cells used in dyr train and evaluation.
"""

import os
import math
import numpy as np
from mindspore import log as logger
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.train.callback import Callback
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR

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
                if (steps_per_epoch != int(name_split[len(name_split)-1])) \
                        or (epoch_num != int(filename[pre_len + index + 1:pre_len + index + 2])):
                    continue
                num = filename[pre_len + 1:pre_len + index]
                if int(num) > max_num:
                    max_num = int(num)
                    load_finetune_checkpoint_path = os.path.join(load_finetune_checkpoint_dir, filename)
    return load_finetune_checkpoint_path


class DynamicRankerLearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for DynamicRanker network.
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(DynamicRankerLearningRate, self).__init__()
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

class MRR():
    """
    Calculate MRR@100 and MRR@10.
    """
    def _mrr(self, gt, pred, val):
        """
        Calculate MRR
        """
        score = 0.0
        for rank, item in enumerate(pred[:val]):
            if item in gt:
                score = 1.0 / (rank + 1.0)
                break
        return score

    def _get_qrels(self, qrels_path):
        """
        Get qrels
        """
        qrels = {}
        with open(qrels_path) as qf:
            for line in qf:
                qid, _, docid, _ = line.strip().split()
                if qid in qrels:
                    qrels[qid].append(docid)
                else:
                    qrels[qid] = [docid]
        return qrels

    def _get_scores(self, scores_path):
        """
        Get scores
        """
        scores = {}
        with open(scores_path) as sf:
            for line in sf:
                qid, docid, score = line.strip().split()
                if qid in scores:
                    scores[qid] += [(docid, float(score))]
                else:
                    scores[qid] = [(docid, float(score))]
        for qid in scores:
            scores[qid] = sorted(scores[qid], key=lambda x: x[1], reverse=True)
        return scores

    def _calc(self, qrels, scores):
        """
        Calculate MRR@100 and MRR@10.
        """
        cn = 0
        mrr100 = []
        mrr10 = []
        for qid in scores:
            if qid in qrels:
                gold_set = set(qrels[qid])
                y = [s[0] for s in scores[qid]]
                mrr100 += [self._mrr(gt=gold_set, pred=y, val=100)]
                mrr10 += [self._mrr(gt=gold_set, pred=y, val=10)]
            else:
                cn += 1
        return mrr100, mrr10

    def accuracy(self, qrels_path, scores_path):
        """
        Calculate MRR@100 and MRR@10.
        Args:
            qrels_path : Path of qrels file.
            score_path : Path of scores file.
        """
        qrels = self._get_qrels(qrels_path)
        scores = self._get_scores(scores_path)
        mrr100, mrr10 = self._calc(qrels, scores)
        print(f"mrr@100:{np.mean(mrr100)}, mrr@10:{np.mean(mrr10)} ")
