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
"""hub config."""
from mindspore.nn import Accuracy
import src.dataset as data
import src.metric as metric
from src.bert_for_finetune import BertCLS
from src.finetune_eval_config import (bert_net_cfg, bert_net_udc_cfg)

TASK_CLASSES = {
        'udc': (data.UDCv1, metric.RecallAtK),
        'dstc2': (data.DSTC2, metric.JointAccuracy),
        'atis_slot': (data.ATIS_DSF, metric.F1Score),
        'atis_intent': (data.ATIS_DID, Accuracy),
        'mrda': (data.MRDA, Accuracy),
        'swda': (data.SwDA, Accuracy)
    }

def set_bert_cfg(task_name):
    """set bert cfg"""
    if task_name == 'udc':
        net_cfg = bert_net_udc_cfg
        eval_net_cfg = bert_net_udc_cfg
        print("use udc_bert_cfg")
    else:
        net_cfg = bert_net_cfg
        eval_net_cfg = bert_net_cfg
    return net_cfg, eval_net_cfg

def create_network(name, *args, **kwargs):
    """set net work which depended it task_name"""
    if name == "dgu":
        if "task_name" in kwargs:
            taskName = kwargs.get("task_name")

        else:
            taskName = "udc"

        if "trainable" in kwargs:
            isTrained = kwargs.get("task_name")

        else:
            isTrained = False

        net_cfg, eval_net_cfg = set_bert_cfg(taskName)
        dataset_class, metric_class = TASK_CLASSES[taskName]
        num_class = dataset_class.num_classes()

        if isTrained:
            # new With loss
            net = BertCLS(net_cfg, True, num_labels=num_class, dropout_prob=0.1)
        else:
            net = BertCLS(eval_net_cfg, False, num_labels=num_class)

        return net
    raise NotImplementedError(f"{name} is not implemented in the repo")
