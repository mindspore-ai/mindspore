# Copyright 2025 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import numpy as np
import math
import os
import tempfile
import json
import re

import mindspore.common.dtype as mstype

from pathlib import Path
from mindspore import Tensor, context, nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops.operations.nn_ops import FlashAttentionScore


def generate_e2edump_json(dump_path, json_file_name, extra_settings_func=None):
    current_dir = Path(__file__).parent
    json_path = current_dir / "test_e2e_statistic_config.json"
    with open(json_path, 'r') as file:
        data = json.load(file)
        data["common_dump_settings"]["path"] = dump_path
        if extra_settings_func is not None:
            extra_settings_func(data)
    with open(json_file_name, 'w') as f:
        json.dump(data, f)


def check_csv_with_regex(file_path, patterns):
    res = [False] * len(patterns)
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            for idx, pattern in enumerate(patterns):
                if pattern.match(line):
                    res[idx] = True
        return res


class FlashAttentionNet(nn.Cell):
    def __init__(self, num_heads, head_dim, dropout_rate=0.0, prev_tockens=65536, next_tockens=65536):
        super(FlashAttentionNet, self).__init__()
        self.keep_prob = 1.0 - dropout_rate
        self.flash_attention = FlashAttentionScore(head_num=num_heads, pre_tokens=prev_tockens,
                                                   next_tokens=next_tockens,
                                                   keep_prob=self.keep_prob,
                                                   scale_value=1.0 /
                                                   math.sqrt(head_dim),
                                                   inner_precise=0,
                                                   input_layout="BNSD")
        self.transpose_key = P.Transpose()

        if self.keep_prob < 1.0:
            self.keep_prob_tensor = Tensor(
                self.keep_prob, dtype=mstype.float16)
            self.drop_gen_mask = P.DropoutGenMask()

    def construct(self, query, key, value, real_shift, attention_mask):
        bsz, head_num, seq_len, _ = query.shape
        key = self.transpose_key(key, (0, 1, 3, 2))
        if self.keep_prob < 1.0:
            drop_mask = F.reshape(self.drop_gen_mask((bsz, head_num, seq_len, seq_len), self.keep_prob_tensor),
                                  ((bsz, head_num, seq_len, seq_len // 8)))
        else:
            drop_mask = None
        attention_mask = F.reshape(attention_mask, (bsz, 1, seq_len, seq_len))
        _, _, _, attention = self.flash_attention(
            query, key, value, real_shift, drop_mask, None, attention_mask, None)
        return attention


class FlashAttentionGradNet(nn.Cell):
    def __init__(self, network):
        super(FlashAttentionGradNet, self).__init__()
        self.network = network
        self.grad = C.GradOperation(get_all=True)

    def construct(self, *inputs):
        gout = self.grad(self.network)(*inputs)
        return gout


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1210.0)
def test_flash_attention_useless_dump():
    """
    Feature: test dump ignore flash attention useless output
    Description: Verify the result of dump result
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    B = 1
    S = 4096
    H = 128
    N = 8
    D = H // N

    def extra_json_settings(data):
        data["e2e_dump_settings"]["stat_calc_mode"] = "device"
        data["e2e_dump_settings"]["enable"] = True

    with tempfile.TemporaryDirectory() as test_dir:
        path = Path(test_dir)
        dump_path = str(path / "dump_data")
        dump_config_path = str(path / "config.json")
        generate_e2edump_json(dump_path, dump_config_path, extra_json_settings)
        try:
            os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
            # Generate inputs
            np.random.seed(1234)
            qv_tensor = Tensor(
                np.random.uniform(-3, 3, (B, N, S, D)), dtype=mstype.float16)
            k_tensor = Tensor(
                np.random.uniform(-3, 3, (B, N, D, S)), dtype=mstype.float16)
            attention_mask = Tensor(np.repeat(np.expand_dims(
                1 - np.tril(np.ones(shape=(S, S))), 0), B, axis=0), dtype=mstype.uint8)
            real_shift = Tensor(
                np.random.uniform(-3, 3, (B, N, S, S)), dtype=mstype.float16)

            fa_net = FlashAttentionNet(num_heads=N, head_dim=D)

            grad_net = FlashAttentionGradNet(fa_net)
            grad_out = grad_net(qv_tensor, k_tensor, qv_tensor, real_shift, attention_mask)
            print(grad_out[0].asnumpy())

            p_flashattention_grads = [re.compile(
                f'^FlashAttentionScoreGrad,.*output,{idx},.*$', re.IGNORECASE) for idx in range(4)]
            p_flashattentions = [re.compile(
                f'^FlashAttentionScore,.*output,{idx},.*$', re.IGNORECASE) for idx in range(4)]
            res = check_csv_with_regex(path / "dump_data/rank_0/Net/0/0/statistic.csv",
                                       p_flashattention_grads+p_flashattentions)
            assert res == [True, True, True, False, True, True, False, True]
        finally:
            del os.environ['MINDSPORE_DUMP_CONFIG']
