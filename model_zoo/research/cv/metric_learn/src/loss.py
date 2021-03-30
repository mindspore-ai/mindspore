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
"""define loss function for network"""
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import operations as P
from mindspore.ops import functional as F

class Softmaxloss(_Loss):
    """Softmaxloss"""
    def __init__(self, sparse=True, smooth_factor=0.1, num_classes=5184):
        super(Softmaxloss, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=sparse, reduction="mean")
    def construct(self, logit, label=None):
        """Tripletloss"""
        if not self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss

class Tripletloss(_Loss):
    """Tripletloss"""
    def __init__(self, margin=0.1):
        super(Tripletloss, self).__init__()
        self.margin = margin
        self.sqrt = P.Sqrt()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.square = P.Square()
        self.div = P.Div()
        self.reshape = P.Reshape()
        self.split = P.Split(1, 3)
        self.relu = nn.ReLU()
        self.expand_dims = P.ExpandDims()
    def construct(self, logit, label=None):
        """Tripletloss c"""
        fea_dim = logit.shape[1]
        input_norm = self.sqrt(self.reduce_sum(self.square(logit), 1))
        logit = self.div(logit, input_norm)
        output = self.reshape(logit, (-1, 3, fea_dim))
        anchor, positive, negative = self.split(output)
        anchor = F.reshape(anchor, (-1, fea_dim))
        positive = self.reshape(positive, (-1, fea_dim))
        negative = self.reshape(negative, (-1, fea_dim))
        a_p = self.square(anchor - positive)
        a_n = self.square(anchor - negative)
        a_p = self.reduce_sum(a_p, 1)
        a_n = self.reduce_sum(a_n, 1)
        loss = a_p - a_n + self.margin
        loss = self.relu(loss)
        return loss

def generate_index(batch_size, samples_each_class):
    """generate_index"""
    a = np.arange(0, batch_size * batch_size, 1)
    a = a.reshape(-1, batch_size)
    #steps = batch_size // samples_each_class
    res = []
    for i in range(batch_size):
        step = i // samples_each_class
        start = step * samples_each_class
        end = (step + 1) * samples_each_class
        p = []
        n = []
        for j, k in enumerate(a[i]):
            if start <= j < end:
                if j == i:
                    p.insert(0, k)
                else:
                    p.append(k)
            else:
                n.append(k)
        comb = p + n
        res += comb
    res = np.array(res).astype(np.int32)
    return res

class Quadrupletloss(_Loss):
    """Quadrupletloss"""
    def __init__(self, train_batch_size=30, samples_each_class=2, margin=0.1):
        super(Quadrupletloss, self).__init__()
        self.margin = margin
        self.samples_each_class = samples_each_class
        self.train_batch_size = train_batch_size
        assert self.train_batch_size % samples_each_class == 0
        self.sqrt = P.Sqrt()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.reduce_sum1 = P.ReduceSum(keep_dims=False)
        self.square = P.Square()
        self.div = P.Div()
        self.reshape = P.Reshape()
        self.relu = nn.ReLU()
        self.reduce_max = P.ReduceMax()
        self.reduce_min = P.ReduceMin()
        self.matmul = nn.MatMul(False, True)
        self.tensoradd = P.TensorAdd()
        #self.tensoradd = P.Add()
        self.assign = P.Assign()
        self.gather = P.GatherV2()
        self.index = generate_index(self.train_batch_size, self.samples_each_class)
        self.index = Tensor(self.index, mstype.int32)
        self.index_var = mindspore.Parameter(Tensor(np.zeros(self.train_batch_size * self.train_batch_size),
                                                    mindspore.int32), name='index_var')
    def construct(self, logit, label=None):
        """Quadrupletloss c"""
        input_norm = self.sqrt(self.reduce_sum(self.square(logit), 1))
        logit = self.div(logit, input_norm)
        margin = self.margin
        feature = self.reshape(logit, (self.train_batch_size, -1))
        ab = self.matmul(feature, feature)
        a2 = self.square(feature)
        a2 = self.reduce_sum1(a2)
        d = self.tensoradd(-2*ab, a2)
        d = self.tensoradd(d, a2)
        d = self.reshape(d, (-1, 1))
        self.index_var = self.assign(self.index_var, self.index)
        d = self.gather(d, self.index_var, 0)
        dd = self.reshape(d, (-1, self.train_batch_size))
        ignore = dd[:, 0 : 1]
        ignore = F.stop_gradient(ignore)
        pos = dd[:, 1 : self.samples_each_class]
        neg = dd[:, self.samples_each_class: self.train_batch_size]
        pos_max = self.reduce_max(pos)
        neg_min = self.reduce_min(neg)
        loss = self.relu(pos_max - neg_min + margin)
        return loss
