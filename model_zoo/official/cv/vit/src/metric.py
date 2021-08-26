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
"""metric"""

import numpy as np

from mindspore.communication.management import GlobalComm
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter


class ClassifyCorrectWithCache(nn.Cell):
    """ClassifyCorrectWithCache"""
    def __init__(self, network, eval_dataset):
        super(ClassifyCorrectWithCache, self).__init__(auto_prefix=False)
        self._network = network
        self.argmax = P.Argmax()
        self.equal = P.Equal()
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()
        self.allreduce = P.AllReduce(P.ReduceOp.SUM, GlobalComm.WORLD_COMM_GROUP)
        self.assign_add = P.AssignAdd()
        self.assign = P.Assign()
        self._correct_num = Parameter(Tensor(0.0, mstype.float32), name="correct_num", requires_grad=False)
        # save data to parameter
        pdata = []
        plabel = []
        step_num = 0
        for batch in eval_dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
            pdata.append(batch["image"])
            plabel.append(batch["label"])
            step_num = step_num + 1
        pdata = Tensor(np.array(pdata), mstype.float32)
        plabel = Tensor(np.array(plabel), mstype.int32)
        self._data = Parameter(pdata, name="pdata", requires_grad=False)
        self._label = Parameter(plabel, name="plabel", requires_grad=False)
        self._step_num = Tensor(step_num, mstype.int32)

    def construct(self, index):
        self._correct_num = 0
        while index < self._step_num:
            data = self._data[index]
            label = self._label[index]
            outputs = self._network(data)
            y_pred = self.argmax(outputs)
            y_pred = self.cast(y_pred, mstype.int32)
            y_correct = self.equal(y_pred, label)
            y_correct = self.cast(y_correct, mstype.float32)
            y_correct_sum = self.reduce_sum(y_correct)
            self._correct_num += y_correct_sum #self.assign(self._correct_num, y_correct_sum)
            index = index + 1
        total_correct = self.allreduce(self._correct_num)
        return total_correct


class ClassifyCorrectCell(nn.Cell):
    """ClassifyCorrectCell"""
    def __init__(self, network):
        super(ClassifyCorrectCell, self).__init__(auto_prefix=False)
        self._network = network
        self.argmax = P.Argmax()
        self.equal = P.Equal()
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()
        self.allreduce = P.AllReduce(P.ReduceOp.SUM, GlobalComm.WORLD_COMM_GROUP)

    def construct(self, data, label):
        outputs = self._network(data)
        y_pred = self.argmax(outputs)
        y_pred = self.cast(y_pred, mstype.int32)
        y_correct = self.equal(y_pred, label)
        y_correct = self.cast(y_correct, mstype.float32)
        y_correct = self.reduce_sum(y_correct)
        total_correct = self.allreduce(y_correct)
        return (total_correct,)


class DistAccuracy(nn.Metric):
    """DistAccuracy"""
    def __init__(self, batch_size, device_num):
        super(DistAccuracy, self).__init__()
        self.clear()
        self.batch_size = batch_size
        self.device_num = device_num

    def clear(self):
        self._correct_num = 0
        self._total_num = 0

    def update(self, *inputs):
        if len(inputs) != 1:
            raise ValueError('Distribute accuracy needs 1 input (y_correct), but got {}'.format(len(inputs)))
        y_correct = self._convert_data(inputs[0])
        self._correct_num += y_correct
        self._total_num += self.batch_size * self.device_num

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError('Accuracy can not be calculated, because the number of samples is 0.')
        return self._correct_num / 50000
