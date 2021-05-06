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
"""Loss and accuracy."""
from mindspore import nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.nn.metrics import Metric


class Loss(nn.Cell):
    """Softmax cross-entropy loss with masking."""
    def __init__(self, mask, weight_decay, param):
        super(Loss, self).__init__(auto_prefix=False)
        self.mask = Tensor(mask)
        self.loss = P.SoftmaxCrossEntropyWithLogits()
        self.one = Tensor(1.0, mstype.float32)
        self.zero = Tensor(0.0, mstype.float32)
        self.mean = P.ReduceMean()
        self.cast = P.Cast()
        self.l2_loss = P.L2Loss()
        self.reduce_sum = P.ReduceSum()
        self.weight_decay = weight_decay
        self.param = param

    def construct(self, preds, label):
        """Calculate loss"""
        param = self.l2_loss(self.param)
        loss = self.weight_decay * param
        preds = self.cast(preds, mstype.float32)
        loss = loss + self.loss(preds, label)[0]
        mask = self.cast(self.mask, mstype.float32)
        mask_reduce = self.mean(mask)
        mask = mask / mask_reduce
        loss = loss * mask
        loss = self.mean(loss)
        return loss

class GCNAccuracy(Metric):
    """
    Accuracy for GCN
    """
    def __init__(self, mask):
        super(GCNAccuracy, self).__init__()
        self.mask = Tensor(mask)
        self.equal = P.Equal()
        self.argmax = P.Argmax()
        self.cast = P.Cast()
        self.mean = P.ReduceMean()
        self.accuracy_all = 0

    def clear(self):
        self.accuracy_all = 0

    def update(self, *inputs):
        preds = self.cast(inputs[1], mstype.float32)
        correct_prediction = self.equal(self.argmax(preds), self.argmax(inputs[0]))
        self.accuracy_all = self.cast(correct_prediction, mstype.float32)
        mask = self.cast(self.mask, mstype.float32)
        mask_reduce = self.mean(mask)
        mask = mask / mask_reduce
        self.accuracy_all *= mask

    def eval(self):
        return float(self.mean(self.accuracy_all).asnumpy())

def apply_eval(eval_param_dict):
    """run Evaluation"""
    model = eval_param_dict["model"]
    dataset = eval_param_dict["dataset"]
    metrics_name = eval_param_dict["metrics_name"]
    eval_score = model.eval(dataset, dataset_sink_mode=False)[metrics_name]
    return eval_score
