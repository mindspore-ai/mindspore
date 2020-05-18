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

"""Component that construct function that compile forward graph."""

from ...components.icomponent import IBuilderComponent
from ...utils.block_util import compile_block, gen_net, create_funcs


class CompileBlockBC(IBuilderComponent):
    """
    Build a function that do mindspore compile.

    Examples:
        'block': BertAttention(batch_size=1,
                               from_tensor_width=1024,
                               to_tensor_width=1024,
                               from_seq_length=128,
                               to_seq_length=128,
                               num_attention_heads=16,
                               size_per_head=64,
                               query_act=None,
                               key_act=None,
                               value_act=None,
                               has_attention_mask=True,
                               attention_probs_dropout_prob=0.0,
                               use_one_hot_embeddings=False,
                               initializer_range=0.02,
                               do_return_2d_tensor=True,
                               use_relative_positions=True,
                               dtype=mstype.float32,
                               compute_type=mstype.float32)
    """

    def __call__(self):
        return create_funcs(self.verification_set, gen_net, compile_block)
