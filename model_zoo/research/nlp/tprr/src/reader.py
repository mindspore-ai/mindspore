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
"""Reader model"""

import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net
from mindspore.ops import BatchMatMul
from mindspore import ops
from mindspore import dtype as mstype
from src.albert import Albert
from src.reader_downstream import Reader_Downstream


dst_type = mstype.float16
dst_type2 = mstype.float32


class Reader(nn.Cell):
    """Reader model"""
    def __init__(self, batch_size, encoder_ck_file, downstream_ck_file):
        """init function"""
        super(Reader, self).__init__(auto_prefix=False)

        self.encoder = Albert(batch_size)
        param_dict = load_checkpoint(encoder_ck_file)
        not_load_params = load_param_into_net(self.encoder, param_dict)
        print(f"reader albert not loaded params: {not_load_params}")

        self.downstream = Reader_Downstream()
        param_dict = load_checkpoint(downstream_ck_file)
        not_load_params = load_param_into_net(self.downstream, param_dict)
        print(f"reader downstream not loaded params: {not_load_params}")

        self.bmm = BatchMatMul()

    def construct(self, input_ids, attn_mask, token_type_ids,
                  context_mask, square_mask, packing_mask, cache_mask,
                  para_start_mapping, sent_end_mapping):
        """construct function"""
        state = self.encoder(input_ids, attn_mask, token_type_ids)

        para_state = self.bmm(ops.Cast()(para_start_mapping, dst_type), ops.Cast()(state, dst_type))  # [B, 2, D]
        sent_state = self.bmm(ops.Cast()(sent_end_mapping, dst_type), ops.Cast()(state, dst_type))  # [B, max_sent, D]

        q_type, start, end, para_logit, sent_logit = self.downstream(ops.Cast()(para_state, dst_type2),
                                                                     ops.Cast()(sent_state, dst_type2),
                                                                     state,
                                                                     context_mask)

        outer = start[:, :, None] + end[:, None]

        outer_mask = cache_mask
        outer_mask = square_mask * outer_mask[None]
        outer = outer - 1e30 * (1 - outer_mask)
        outer = outer - 1e30 * packing_mask[:, :, None]
        max_row = ops.ReduceMax()(outer, 2)
        y1 = ops.Argmax()(max_row)
        max_col = ops.ReduceMax()(outer, 1)
        y2 = ops.Argmax()(max_col)

        return start, end, q_type, para_logit, sent_logit, y1, y2
