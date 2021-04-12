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
"""Reranker Model"""

import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net
from src.albert import Albert
from src.rerank_downstream import Rerank_Downstream


class Reranker(nn.Cell):
    """Reranker model"""
    def __init__(self, batch_size, encoder_ck_file, downstream_ck_file):
        """init function"""
        super(Reranker, self).__init__(auto_prefix=False)

        self.encoder = Albert(batch_size)
        param_dict = load_checkpoint(encoder_ck_file)
        not_load_params_1 = load_param_into_net(self.encoder, param_dict)
        print(f"re-ranker albert not loaded params: {not_load_params_1}")

        self.no_answer_mlp = Rerank_Downstream()
        param_dict = load_checkpoint(downstream_ck_file)
        not_load_params_2 = load_param_into_net(self.no_answer_mlp, param_dict)
        print(f"re-ranker downstream not loaded params: {not_load_params_2}")

    def construct(self, input_ids, attn_mask, token_type_ids):
        """construct function"""
        state = self.encoder(input_ids, attn_mask, token_type_ids)
        state = state[:, 0, :]

        no_answer = self.no_answer_mlp(state)
        return no_answer
