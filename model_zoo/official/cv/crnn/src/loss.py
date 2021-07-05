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
"""CTC Loss."""
import numpy as np
from mindspore.nn.loss.loss import LossBase
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P


class CTCLoss(LossBase):
    """
     CTCLoss definition

     Args:
        max_sequence_length(int): max number of sequence length. For text images, the value is equal to image
        width
        max_label_length(int): max number of label length for each input.
        batch_size(int): batch size of input logits
     """

    def __init__(self, max_sequence_length, max_label_length, batch_size):
        super(CTCLoss, self).__init__()
        self.sequence_length = Parameter(Tensor(np.array([max_sequence_length] * batch_size), mstype.int32),
                                         name="sequence_length")
        labels_indices = []
        for i in range(batch_size):
            for j in range(max_label_length):
                labels_indices.append([i, j])
        self.labels_indices = Parameter(Tensor(np.array(labels_indices), mstype.int64), name="labels_indices")
        self.reshape = P.Reshape()
        self.ctc_loss = P.CTCLoss(ctc_merge_repeated=True)

    def construct(self, logit, label):
        labels_values = self.reshape(label, (-1,))
        loss, _ = self.ctc_loss(logit, self.labels_indices, labels_values, self.sequence_length)
        return loss
