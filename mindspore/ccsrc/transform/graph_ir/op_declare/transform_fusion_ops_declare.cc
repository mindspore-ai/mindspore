/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "transform/graph_ir/op_declare/transform_fusion_ops_declare.h"
#include <vector>
#include <string>
#include "ops/fusion/flash_attention.h"
#include "ops/fusion/matmul_allreduce.h"

namespace mindspore::transform {
// KVCacheMgr
INPUT_MAP(KVCacheMgr) = {{1, INPUT_DESC(past)}, {2, INPUT_DESC(cur)}, {3, INPUT_DESC(index)}};
ATTR_MAP(KVCacheMgr) = EMPTY_ATTR_MAP;
OUTPUT_MAP(KVCacheMgr) = {{0, OUTPUT_DESC(past)}};
REG_ADPT_DESC(KVCacheMgr, "KVCacheMgr", ADPT_DESC(KVCacheMgr))

// DecoderKVCache
INPUT_MAP(DecoderKvCache) = {{1, INPUT_DESC(cache)},          {2, INPUT_DESC(update)},
                             {3, INPUT_DESC(valid_seq_len)},  {4, INPUT_DESC(batch_index)},
                             {5, INPUT_DESC(seq_len_axis)},   {6, INPUT_DESC(new_max_seq_len)},
                             {7, INPUT_DESC(cur_max_seq_len)}};
ATTR_MAP(DecoderKvCache) = EMPTY_ATTR_MAP;
OUTPUT_MAP(DecoderKvCache) = {{0, OUTPUT_DESC(out)}};
REG_ADPT_DESC(DecoderKvCache, "DecoderKVCache", ADPT_DESC(DecoderKvCache))

// PromptKVCache
INPUT_MAP(PromptKvCache) = {{1, INPUT_DESC(cache)},          {2, INPUT_DESC(update)},
                            {3, INPUT_DESC(valid_seq_len)},  {4, INPUT_DESC(batch_index)},
                            {5, INPUT_DESC(seq_len_axis)},   {6, INPUT_DESC(new_max_seq_len)},
                            {7, INPUT_DESC(cur_max_seq_len)}};
ATTR_MAP(PromptKvCache) = EMPTY_ATTR_MAP;
OUTPUT_MAP(PromptKvCache) = {{0, OUTPUT_DESC(out)}};
REG_ADPT_DESC(PromptKvCache, "PromptKVCache", ADPT_DESC(PromptKvCache))

// FlashAttention
INPUT_MAP(FlashAttention) = {
  {1, INPUT_DESC(q)}, {2, INPUT_DESC(k)}, {3, INPUT_DESC(v)}, {4, INPUT_DESC(attention_mask)}};
ATTR_MAP(FlashAttention) = EMPTY_ATTR_MAP;
OUTPUT_MAP(FlashAttention) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FlashAttention, ops::kNameFlashAttention, ADPT_DESC(FlashAttention))

// FFN
INPUT_MAP(FFN) = {
  {1, INPUT_DESC(x)},          {2, INPUT_DESC(weight1)},    {3, INPUT_DESC(weight2)}, {4, INPUT_DESC(expert_tokens)},
  {5, INPUT_DESC(bias1)},      {6, INPUT_DESC(bias2)},      {7, INPUT_DESC(scale)},   {8, INPUT_DESC(offset)},
  {9, INPUT_DESC(deq_scale1)}, {10, INPUT_DESC(deq_scale2)}};
ATTR_MAP(FFN) = {{"activation", ATTR_DESC(activation, AnyTraits<string>())},
                 {"inner_precise", ATTR_DESC(inner_precise, AnyTraits<int64_t>())}};
OUTPUT_MAP(FFN) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FFN, kNameFFN, ADPT_DESC(FFN))

// MatMulAllReduce
INPUT_MAP(MatmulAllReduce) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(bias)}};
OUTPUT_MAP(MatmulAllReduce) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(MatmulAllReduce) = {{"group", ATTR_DESC(group, AnyTraits<std::string>())},
                             {"op", ATTR_DESC(reduce_op, AnyTraits<std::string>())},
                             {"transpose_a", ATTR_DESC(is_trans_a, AnyTraits<bool>())},
                             {"transpose_b", ATTR_DESC(is_trans_b, AnyTraits<bool>())},
                             {"comm_reuse", ATTR_DESC(comm_turn, AnyTraits<int>())}};
REG_ADPT_DESC(MatMulAllReduce, kNameMatMulAllReduce, ADPT_DESC(MatmulAllReduce))
}  // namespace mindspore::transform
