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

import random
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.dataset as de
from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore.train.serialization import export
from tests.st.networks.models.bert.src.bert_model import BertModel, BertConfig

bert_net_cfg = BertConfig(
    batch_size=2,
    seq_length=32,
    vocab_size=21128,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=False,
    input_mask_from_dataset=True,
    token_type_ids_from_dataset=True,
    dtype=mstype.float32,
    compute_type=mstype.float16
)

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

class AddNet(nn.Cell):
    def __init__(self):
        super(AddNet, self).__init__()
        self.add = P.TensorAdd()

    def construct(self, x_, y_):
        return self.add(x_, y_)

def export_add_model():
    net = AddNet()
    x = np.ones(4).astype(np.float32)
    y = np.ones(4).astype(np.float32)
    export(net, Tensor(x), Tensor(y), file_name='add.mindir', file_format='MINDIR')

def export_bert_model():
    input_ids = np.random.randint(0, 1000, size=(2, 32), dtype=np.int32)
    segment_ids = np.zeros((2, 32), dtype=np.int32)
    input_mask = np.zeros((2, 32), dtype=np.int32)
    net = BertModel(bert_net_cfg, False)
    export(net, Tensor(input_ids), Tensor(segment_ids), Tensor(input_mask),
           file_name='bert.mindir', file_format='MINDIR')

if __name__ == '__main__':
    export_add_model()
    export_bert_model()
