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
"""GRU Infer cell"""
import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype
from src.config import config

class GRUInferCell(nn.Cell):
    '''
    GRU infer consturction

    Args:
        network: gru network
    '''
    def __init__(self, network):
        super(GRUInferCell, self).__init__(auto_prefix=False)
        self.network = network
        self.argmax = P.ArgMaxWithValue(axis=2)
        self.transpose = P.Transpose()
        self.teacher_force = Tensor(np.zeros((config.eval_batch_size)), mstype.bool_)
    def construct(self,
                  encoder_inputs,
                  decoder_inputs):
        predict_probs = self.network(encoder_inputs, decoder_inputs, self.teacher_force)
        predict_probs = self.transpose(predict_probs, (1, 0, 2))
        predict_ids, _ = self.argmax(predict_probs)
        return predict_ids
