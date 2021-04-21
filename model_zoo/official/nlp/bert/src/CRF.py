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

'''
CRF script.
'''

import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype

class CRF(nn.Cell):
    '''
    Conditional Random Field
    Args:
        tag_to_index: The dict for tag to index mapping with extra "<START>" and "<STOP>"sign.
        batch_size: Batch size, i.e., the length of the first dimension.
        seq_length: Sequence length, i.e., the length of the second dimension.
        is_training: Specifies whether to use training mode.
    Returns:
        Training mode: Tensor, total loss.
        Evaluation mode: Tuple, the index for each step with the highest score; Tuple, the index for the last
        step with the highest score.
    '''
    def __init__(self, tag_to_index, batch_size=1, seq_length=128, is_training=True):

        super(CRF, self).__init__()
        self.target_size = len(tag_to_index)
        self.is_training = is_training
        self.tag_to_index = tag_to_index
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.START_VALUE = Tensor(self.target_size-2, dtype=mstype.int32)
        self.STOP_VALUE = Tensor(self.target_size-1, dtype=mstype.int32)
        transitions = np.random.normal(size=(self.target_size, self.target_size)).astype(np.float32)
        transitions[tag_to_index[self.START_TAG], :] = -10000
        transitions[:, tag_to_index[self.STOP_TAG]] = -10000
        self.transitions = Parameter(Tensor(transitions))
        self.cat = P.Concat(axis=-1)
        self.argmax = P.ArgMaxWithValue(axis=-1)
        self.log = P.Log()
        self.exp = P.Exp()
        self.sum = P.ReduceSum()
        self.tile = P.Tile()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.reshape = P.Reshape()
        self.expand = P.ExpandDims()
        self.mean = P.ReduceMean()
        init_alphas = np.ones(shape=(self.batch_size, self.target_size)) * -10000.0
        init_alphas[:, self.tag_to_index[self.START_TAG]] = 0.
        self.init_alphas = Tensor(init_alphas, dtype=mstype.float32)
        self.cast = P.Cast()
        self.reduce_max = P.ReduceMax(keep_dims=True)
        self.on_value = Tensor(1.0, dtype=mstype.float32)
        self.off_value = Tensor(0.0, dtype=mstype.float32)
        self.onehot = P.OneHot()

    def log_sum_exp(self, logits):
        '''
        Compute the log_sum_exp score for Normalization factor.
        '''
        max_score = self.reduce_max(logits, -1)  #16 5 5
        score = self.log(self.reduce_sum(self.exp(logits - max_score), -1))
        score = max_score + score
        return score

    def _realpath_score(self, features, label):
        '''
        Compute the emission and transition score for the real path.
        '''
        label = label * 1
        concat_A = self.tile(self.reshape(self.START_VALUE, (1,)), (self.batch_size,))
        concat_A = self.reshape(concat_A, (self.batch_size, 1))
        labels = self.cat((concat_A, label))
        onehot_label = self.onehot(label, self.target_size, self.on_value, self.off_value)
        emits = features * onehot_label
        labels = self.onehot(labels, self.target_size, self.on_value, self.off_value)
        label1 = labels[:, 1:, :]
        label2 = labels[:, :self.seq_length, :]
        label1 = self.expand(label1, 3)
        label2 = self.expand(label2, 2)
        label_trans = label1 * label2
        transitions = self.expand(self.expand(self.transitions, 0), 0)
        trans = transitions * label_trans
        score = self.sum(emits, (1, 2)) + self.sum(trans, (1, 2, 3))
        stop_value_index = labels[:, (self.seq_length-1):self.seq_length, :]
        stop_value = self.transitions[(self.target_size-1):self.target_size, :]
        stop_score = stop_value * self.reshape(stop_value_index, (self.batch_size, self.target_size))
        score = score + self.sum(stop_score, 1)
        score = self.reshape(score, (self.batch_size, -1))
        return score

    def _normalization_factor(self, features):
        '''
        Compute the total score for all the paths.
        '''
        forward_var = self.init_alphas
        forward_var = self.expand(forward_var, 1)
        for idx in range(self.seq_length):
            feat = features[:, idx:(idx+1), :]
            emit_score = self.reshape(feat, (self.batch_size, self.target_size, 1))
            next_tag_var = emit_score + self.transitions + forward_var
            forward_var = self.log_sum_exp(next_tag_var)
            forward_var = self.reshape(forward_var, (self.batch_size, 1, self.target_size))
        terminal_var = forward_var + self.reshape(self.transitions[(self.target_size-1):self.target_size, :], (1, -1))
        alpha = self.log_sum_exp(terminal_var)
        alpha = self.reshape(alpha, (self.batch_size, -1))
        return alpha

    def _decoder(self, features):
        '''
        Viterbi decode for evaluation.
        '''
        backpointers = ()
        forward_var = self.init_alphas
        for idx in range(self.seq_length):
            feat = features[:, idx:(idx+1), :]
            feat = self.reshape(feat, (self.batch_size, self.target_size))
            bptrs_t = ()

            next_tag_var = self.expand(forward_var, 1) + self.transitions
            best_tag_id, best_tag_value = self.argmax(next_tag_var)
            bptrs_t += (best_tag_id,)
            forward_var = best_tag_value + feat

            backpointers += (bptrs_t,)
        terminal_var = forward_var + self.reshape(self.transitions[(self.target_size-1):self.target_size, :], (1, -1))
        best_tag_id, _ = self.argmax(terminal_var)
        return backpointers, best_tag_id

    def construct(self, features, label):
        if self.is_training:
            forward_score = self._normalization_factor(features)
            gold_score = self._realpath_score(features, label)
            return_value = self.mean(forward_score - gold_score)
        else:
            path_list, tag = self._decoder(features)
            return_value = path_list, tag
        return return_value

def postprocess(backpointers, best_tag_id):
    '''
    Do postprocess
    '''
    best_tag_id = best_tag_id.asnumpy()
    batch_size = len(best_tag_id)
    best_path = []
    for i in range(batch_size):
        best_path.append([])
        best_local_id = best_tag_id[i]
        best_path[-1].append(best_local_id)
        for bptrs_t in reversed(backpointers):
            bptrs_t = bptrs_t[0].asnumpy()
            local_idx = bptrs_t[i]
            best_local_id = local_idx[best_local_id]
            best_path[-1].append(best_local_id)
        # Pop off the start tag (we dont want to return that to the caller)
        best_path[-1].pop()
        best_path[-1].reverse()
    return best_path
