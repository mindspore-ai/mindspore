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

"""
Bert evaluation script.
"""

import os
import argparse
import math
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.common.tensor import Tensor
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.evaluation_config import cfg, bert_net_cfg
from src.utils import BertNER, BertCLS, BertReg
from src.CRF import postprocess
from src.cluener_evaluation import submit
from src.finetune_config import tag_to_index


class Accuracy():
    """
    calculate accuracy
    """
    def __init__(self):
        self.acc_num = 0
        self.total_num = 0

    def update(self, logits, labels):
        """
        Update accuracy
        """
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logit_id = np.argmax(logits, axis=-1)
        self.acc_num += np.sum(labels == logit_id)
        self.total_num += len(labels)
        print("=========================accuracy is ", self.acc_num / self.total_num)


class F1():
    """
    calculate F1 score
    """
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def update(self, logits, labels):
        """
        update F1 score
        """
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        if cfg.use_crf:
            backpointers, best_tag_id = logits
            best_path = postprocess(backpointers, best_tag_id)
            logit_id = []
            for ele in best_path:
                logit_id.extend(ele)
        else:
            logits = logits.asnumpy()
            logit_id = np.argmax(logits, axis=-1)
            logit_id = np.reshape(logit_id, -1)
        pos_eva = np.isin(logit_id, [i for i in range(1, cfg.num_labels)])
        pos_label = np.isin(labels, [i for i in range(1, cfg.num_labels)])
        self.TP += np.sum(pos_eva&pos_label)
        self.FP += np.sum(pos_eva&(~pos_label))
        self.FN += np.sum((~pos_eva)&pos_label)


class MCC():
    """
    Calculate Matthews Correlation Coefficient.
    """
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0

    def update(self, logits, labels):
        """
        Update MCC score
        """
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        labels = labels.astype(np.bool)
        logits = logits.asnumpy()
        logit_id = np.argmax(logits, axis=-1)
        logit_id = np.reshape(logit_id, -1)
        logit_id = logit_id.astype(np.bool)
        ornot = logit_id ^ labels

        self.TP += (~ornot & labels).sum()
        self.FP += (ornot & ~labels).sum()
        self.FN += (ornot & labels).sum()
        self.TN += (~ornot & ~labels).sum()


class Spearman_Correlation():
    """
    calculate Spearman Correlation coefficient
    """
    def __init__(self):
        self.label = []
        self.logit = []

    def update(self, logits, labels):
        """
        Update Spearman Correlation
        """
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logits = np.reshape(logits, -1)
        self.label.append(labels)
        self.logit.append(logits)

    def cal(self):
        """
        Calculate Spearman Correlation
        """
        label = np.concatenate(self.label)
        logit = np.concatenate(self.logit)
        sort_label = label.argsort()[::-1]
        sort_logit = logit.argsort()[::-1]
        n = len(label)
        d_acc = 0
        for i in range(n):
            d = np.where(sort_label == i)[0] - np.where(sort_logit == i)[0]
            d_acc += d**2
        ps = 1 - 6*d_acc/n/(n**2-1)
        return ps


def get_dataset(batch_size=1, repeat_count=1, distribute_file=''):
    """
    get dataset
    """
    _ = distribute_file

    ds = de.TFRecordDataset([cfg.data_file], cfg.schema_file, columns_list=["input_ids", "input_mask",
                                                                            "segment_ids", "label_ids"])
    type_cast_op = C.TypeCast(mstype.int32)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    if cfg.task == "Regression":
        type_cast_op_float = C.TypeCast(mstype.float32)
        ds = ds.map(input_columns="label_ids", operations=type_cast_op_float)
    else:
        ds = ds.map(input_columns="label_ids", operations=type_cast_op)
    ds = ds.repeat(repeat_count)

    # apply shuffle operation
    buffer_size = 960
    ds = ds.shuffle(buffer_size=buffer_size)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def bert_predict(Evaluation):
    """
    prediction function
    """
    target = args_opt.device_target
    if target == "Ascend":
        devid = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=devid)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")
    dataset = get_dataset(bert_net_cfg.batch_size, 1)
    if cfg.use_crf:
        net_for_pretraining = Evaluation(bert_net_cfg, False, num_labels=len(tag_to_index), use_crf=True,
                                         tag_to_index=tag_to_index, dropout_prob=0.0)
    else:
        net_for_pretraining = Evaluation(bert_net_cfg, False, num_labels)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(cfg.finetune_ckpt)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)
    return model, dataset

def test_eval():
    """
    evaluation function
    """
    if cfg.task == "SeqLabeling":
        task_type = BertNER
    elif cfg.task == "Regression":
        task_type = BertReg
    elif cfg.task == "Classification":
        task_type = BertCLS
    elif cfg.task == "COLA":
        task_type = BertCLS
    else:
        raise ValueError("Task not supported.")
    model, dataset = bert_predict(task_type)

    if cfg.clue_benchmark:
        submit(model, cfg.data_file, bert_net_cfg.seq_length)
    else:
        if cfg.task == "SeqLabeling":
            callback = F1()
        elif cfg.task == "COLA":
            callback = MCC()
        elif cfg.task == "Regression":
            callback = Spearman_Correlation()
        else:
            callback = Accuracy()

        columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
        for data in dataset.create_dict_iterator():
            input_data = []
            for i in columns_list:
                input_data.append(Tensor(data[i]))
            input_ids, input_mask, token_type_id, label_ids = input_data
            logits = model.predict(input_ids, input_mask, token_type_id, label_ids)
            callback.update(logits, label_ids)
        print("==============================================================")
        if cfg.task == "SeqLabeling":
            print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
            print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
            print("F1 {:.6f} ".format(2*callback.TP / (2*callback.TP + callback.FP + callback.FN)))
        elif cfg.task == "COLA":
            TP = callback.TP
            TN = callback.TN
            FP = callback.FP
            FN = callback.FN
            mcc = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
            print("MCC: {:.6f}".format(mcc))
        elif cfg.task == "Regression":
            print("Spearman Correlation is {:.6f}".format(callback.cal()[0]))
        else:
            print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                      callback.acc_num / callback.total_num))
        print("==============================================================")

parser = argparse.ArgumentParser(description='Bert eval')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
args_opt = parser.parse_args()
if __name__ == "__main__":
    num_labels = cfg.num_labels
    test_eval()
