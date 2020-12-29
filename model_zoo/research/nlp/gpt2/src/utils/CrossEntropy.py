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
"""Calculate Cross Entropy With Mask"""
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.nn as nn


class CrossEntropyCalculationWithMask(nn.Cell):
    """
    Cross Entropy loss
    """

    def __init__(self, is_training=None, num_labels=None, config=None):
        super(CrossEntropyCalculationWithMask, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.is_training = is_training
        self.num_labels = num_labels
        if config is not None:
            # for PPL calculation in evaluation
            self.input_mask_length = Tensor(config.batch_size * (config.seq_length - 1), mstype.float32)

    def construct(self, logits, label_ids, input_mask=None):
        """
        Calculate loss

        Args:
            logits (Tensor): the probability distribution over vocabulary.
            label_ids (Tensor): the indices of input sequence tokens in the vocabulary.
            input_mask (Tensor): input sentences padding mask, where 0 indicates padding position.

        Returns:
            return_value (Tensor, mstype.float32): if is_training is False, directly return the logits, otherwise,
                                                   return the computed loss.
        """

        # logits [batch * (seq_length-1), vocab_size]   label_ids [batch, seq_length-1]
        if self.is_training:
            label_ids = self.reshape(label_ids, self.last_idx)  # label_ids [batch * (seq_length-1)]
            one_hot_labels = self.onehot(label_ids, self.num_labels, self.on_value,
                                         self.off_value)  # [batch * (seq_length-1), vocab_size]
            per_example_loss = self.neg(
                self.reduce_sum(one_hot_labels * logits, self.last_idx))  # [batch * (seq_length-1)]

            # for PPL calculation in evaluation
            if input_mask is not None:
                input_mask = self.cast(self.reshape(input_mask, self.last_idx),
                                       mstype.float32)  # [batch * (seq_length-1)]

                valid_loss_sum = self.reduce_sum(input_mask * per_example_loss, ())
                valid_element_sum = self.reduce_sum(input_mask, ()) + self.cast(F.tuple_to_array((1e-5,)),
                                                                                mstype.float32)
                loss = valid_loss_sum / valid_element_sum
            else:
                loss = self.reduce_mean(per_example_loss, self.last_idx)  # a number
            return_value = self.cast(loss, mstype.float32)
        else:
            return_value = logits * 1.0  # [batch * (seq_length-1), vocab_size]

        return return_value
