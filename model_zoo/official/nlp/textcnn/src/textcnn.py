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
"""TextCNN"""

import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.nn.cell import Cell
import mindspore.ops.functional as F
import mindspore


class SoftmaxCrossEntropyExpand(Cell):
    r"""
    Computes softmax cross entropy between logits and labels. Implemented by expanded formula.

    This is a wrapper of several functions.

    .. math::
        \ell(x_i, t_i) = -log\left(\frac{\exp(x_{t_i})}{\sum_j \exp(x_j)}\right),
    where :math:`x_i` is a 1D score Tensor, :math:`t_i` is the target class.

    Note:
        When argument sparse is set to True, the format of label is the index
        range from :math:`0` to :math:`C - 1` instead of one-hot vectors.

    Args:
        sparse(bool): Specifies whether labels use sparse format or not. Default: False.

    Inputs:
        - **input_data** (Tensor) - Tensor of shape :math:`(x_1, x_2, ..., x_R)`.
        - **label** (Tensor) - Tensor of shape :math:`(y_1, y_2, ..., y_S)`.

    Outputs:
        Tensor, a scalar tensor including the mean loss.

    Examples:
        >>> loss = nn.SoftmaxCrossEntropyExpand(sparse=True)
        >>> input_data = Tensor(np.ones([64, 512]), dtype=mindspore.float32)
        >>> label = Tensor(np.ones([64]), dtype=mindspore.int32)
        >>> loss(input_data, label)
    """
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = P.Exp()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)
        self.div = P.Div()
        self.log = P.Log()
        self.sum_cross_entropy = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.cast = P.Cast()
        self.reduce_mean = P.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.reduce_max = P.ReduceMax(keep_dims=True)
        self.sub = P.Sub()

    def construct(self, logit, label):
        """
        construct
        """
        logit_max = self.reduce_max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.reduce_sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)

        softmax_result_log = self.log(softmax_result)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(F.scalar_to_array(-1.0), loss)
        loss = self.reduce_mean(loss, -1)

        return loss

def make_conv_layer(kernel_size):
    return nn.Conv2d(in_channels=1, out_channels=96, kernel_size=kernel_size, padding=1,
                     pad_mode="pad", has_bias=True)


class TextCNN(nn.Cell):
    """
    TextCNN architecture
    """
    def __init__(self, vocab_len, word_len, num_classes, vec_length, embedding_table='uniform'):
        super(TextCNN, self).__init__()
        self.vec_length = vec_length
        self.word_len = word_len
        self.num_classes = num_classes

        self.unsqueeze = P.ExpandDims()
        self.embedding = nn.Embedding(vocab_len, self.vec_length, embedding_table=embedding_table)

        self.slice = P.Slice()
        self.layer1 = self.make_layer(kernel_height=3)
        self.layer2 = self.make_layer(kernel_height=4)
        self.layer3 = self.make_layer(kernel_height=5)

        self.concat = P.Concat(1)

        self.fc = nn.Dense(96*3, self.num_classes)
        self.drop = nn.Dropout(keep_prob=0.5)
        self.print = P.Print()
        self.reducemax = P.ReduceMax(keep_dims=False)

    def make_layer(self, kernel_height):
        return nn.SequentialCell(
            [
                make_conv_layer((kernel_height, self.vec_length)), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(self.word_len-kernel_height+1, 1)),
            ]
        )

    def construct(self, x):
        """
        construct
        """
        x = self.unsqueeze(x, 1)
        x = self.embedding(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)

        x1 = self.reducemax(x1, (2, 3))
        x2 = self.reducemax(x2, (2, 3))
        x3 = self.reducemax(x3, (2, 3))

        x = self.concat((x1, x2, x3))
        x = self.drop(x)
        x = self.fc(x)
        return x
