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

#include "transform/graph_ir/op_declare/fusion_ops_declare.h"
#include <vector>
#include <string>

namespace mindspore::transform {
// PromptFlashAttention
INPUT_MAP(PromptFlashAttention) = {
  {1, INPUT_DESC(query)},
  {2, INPUT_DESC(key)},
  {3, INPUT_DESC(value)},
  {4, INPUT_DESC(atten_mask)},             // optional input
  {5, INPUT_DESC(actual_seq_lengths)},     // optional input
  {6, INPUT_DESC(actual_seq_lengths_kv)},  // optional input
  {7, INPUT_DESC(pse_shift)},              // optional input
  {8, INPUT_DESC(deq_scale1)},             // optional input
  {9, INPUT_DESC(quant_scale1)},           // optional input
  {10, INPUT_DESC(deq_scale2)},            // optional input
  {11, INPUT_DESC(quant_scale2)},          // optional input
  {12, INPUT_DESC(quant_offset2)},         // optional input
};
ATTR_MAP(PromptFlashAttention) = {{"num_heads", ATTR_DESC(num_heads, AnyTraits<int64_t>())},
                                  {"pre_tokens", ATTR_DESC(pre_tokens, AnyTraits<int64_t>())},
                                  {"next_tokens", ATTR_DESC(next_tokens, AnyTraits<int64_t>())},
                                  {"input_layout", ATTR_DESC(input_layout, AnyTraits<std::string>())},
                                  {"num_key_value_heads", ATTR_DESC(num_key_value_heads, AnyTraits<int64_t>())},
                                  {"scale_value", ATTR_DESC(scale_value, AnyTraits<float>())}};
OUTPUT_MAP(PromptFlashAttention) = {{0, OUTPUT_DESC(attention_out)}};
REG_ADPT_DESC(PromptFlashAttention, "PromptFlashAttention", ADPT_DESC(PromptFlashAttention))

// IncreFlashAttention
INPUT_MAP(IncreFlashAttention) = {{1, INPUT_DESC(query)},
                                  {4, INPUT_DESC(atten_mask)},
                                  {5, INPUT_DESC(actual_seq_lengths)},
                                  {6, INPUT_DESC(padding_mask)},
                                  {7, INPUT_DESC(dequant_scale1)},
                                  {8, INPUT_DESC(quant_scale1)},
                                  {9, INPUT_DESC(dequant_scale2)},
                                  {10, INPUT_DESC(quant_scale2)},
                                  {11, INPUT_DESC(quant_offset2)},
                                  {12, INPUT_DESC(antiquant_scale)},
                                  {13, INPUT_DESC(antiquant_offset)},
                                  {14, INPUT_DESC(block_table)}};
DYN_INPUT_MAP(IncreFlashAttention) = {{2, DYN_INPUT_DESC(key)}, {3, DYN_INPUT_DESC(value)}};
ATTR_MAP(IncreFlashAttention) = {{"num_heads", ATTR_DESC(num_heads, AnyTraits<int64_t>())},
                                 {"input_layout", ATTR_DESC(input_layout, AnyTraits<std::string>())},
                                 {"scale_value", ATTR_DESC(scale_value, AnyTraits<float>())},
                                 {"num_key_value_heads", ATTR_DESC(num_key_value_heads, AnyTraits<int64_t>())},
                                 {"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())},
                                 {"inner_precise", ATTR_DESC(inner_precise, AnyTraits<int64_t>())}};
OUTPUT_MAP(IncreFlashAttention) = {{0, OUTPUT_DESC(attention_out)}};
REG_ADPT_DESC(IncreFlashAttention, "IncreFlashAttention", ADPT_DESC(IncreFlashAttention))

// FlashAttentionScore
INPUT_MAP(FlashAttentionScore) = {
  {1, INPUT_DESC(query)},     {2, INPUT_DESC(key)},        {3, INPUT_DESC(value)},        {4, INPUT_DESC(atten_mask)},
  {5, INPUT_DESC(drop_mask)}, {6, INPUT_DESC(real_shift)}, {7, INPUT_DESC(padding_mask)}, {8, INPUT_DESC(prefix)}};
ATTR_MAP(FlashAttentionScore) = {
  {"scale_value", ATTR_DESC(scale_value, AnyTraits<float>())},
  {"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())},
  {"pre_tokens", ATTR_DESC(pre_tockens, AnyTraits<int64_t>())},
  {"next_tokens", ATTR_DESC(next_tockens, AnyTraits<int64_t>())},
  {"head_num", ATTR_DESC(head_num, AnyTraits<int64_t>())},
  {"inner_precise", ATTR_DESC(inner_precise, AnyTraits<int64_t>())},
  {"input_layout", ATTR_DESC(input_layout, AnyTraits<std::string>())},
  {"sparse_mode", ATTR_DESC(sparse_mode, AnyTraits<int64_t>())},
};
OUTPUT_MAP(FlashAttentionScore) = {
  {0, OUTPUT_DESC(attention_out)}, {1, OUTPUT_DESC(softmax_max)}, {2, OUTPUT_DESC(softmax_sum)}};
REG_ADPT_DESC(FlashAttentionScore, kNameFlashAttentionScore, ADPT_DESC(FlashAttentionScore))

// FlashAttentionScoreGrad
INPUT_MAP(FlashAttentionScoreGrad) = {{1, INPUT_DESC(query)},         {2, INPUT_DESC(key)},
                                      {3, INPUT_DESC(value)},         {4, INPUT_DESC(atten_mask)},
                                      {5, INPUT_DESC(attention_in)},  {6, INPUT_DESC(softmax_max)},
                                      {7, INPUT_DESC(softmax_sum)},   {8, INPUT_DESC(dy)},
                                      {9, INPUT_DESC(drop_mask)},     {10, INPUT_DESC(pse_shift)},
                                      {11, INPUT_DESC(padding_mask)}, {12, INPUT_DESC(softmax_in)},
                                      {13, INPUT_DESC(prefix)}};
ATTR_MAP(FlashAttentionScoreGrad) = {
  {"scale_value", ATTR_DESC(scale_value, AnyTraits<float>())},
  {"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())},
  {"pre_tokens", ATTR_DESC(pre_tockens, AnyTraits<int64_t>())},
  {"next_tokens", ATTR_DESC(next_tockens, AnyTraits<int64_t>())},
  {"head_num", ATTR_DESC(head_num, AnyTraits<int64_t>())},
  {"inner_precise", ATTR_DESC(inner_precise, AnyTraits<int64_t>())},
  {"input_layout", ATTR_DESC(input_layout, AnyTraits<std::string>())},
  {"sparse_mode", ATTR_DESC(sparse_mode, AnyTraits<int64_t>())},
};
OUTPUT_MAP(FlashAttentionScoreGrad) = {{0, OUTPUT_DESC(dq)}, {1, OUTPUT_DESC(dk)}, {2, OUTPUT_DESC(dv)}};
REG_ADPT_DESC(FlashAttentionScoreGrad, kNameFlashAttentionScoreGrad, ADPT_DESC(FlashAttentionScoreGrad))

// MatmulReduceScatter
INPUT_MAP(MatmulReduceScatter) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(bias)}};
ATTR_MAP(MatmulReduceScatter) = {{"group", ATTR_DESC(group, AnyTraits<std::string>())},
                                 {"reduce_op", ATTR_DESC(reduce_op, AnyTraits<std::string>())},
                                 {"is_trans_a", ATTR_DESC(is_trans_a, AnyTraits<bool>())},
                                 {"is_trans_b", ATTR_DESC(is_trans_b, AnyTraits<bool>())},
                                 {"comm_turn", ATTR_DESC(comm_turn, AnyTraits<int64_t>())}};
OUTPUT_MAP(MatmulReduceScatter) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatmulReduceScatter, kNameMatmulReduceScatter, ADPT_DESC(MatmulReduceScatter))

// AllGatherMatmul
INPUT_MAP(AllGatherMatmul) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(bias)}};
ATTR_MAP(AllGatherMatmul) = {{"group", ATTR_DESC(group, AnyTraits<std::string>())},
                             {"is_trans_a", ATTR_DESC(is_trans_a, AnyTraits<bool>())},
                             {"is_trans_b", ATTR_DESC(is_trans_b, AnyTraits<bool>())},
                             {"gather_index", ATTR_DESC(gather_index, AnyTraits<int64_t>())},
                             {"comm_turn", ATTR_DESC(comm_turn, AnyTraits<int64_t>())}};
OUTPUT_MAP(AllGatherMatmul) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(gather_out)}};
REG_ADPT_DESC(AllGatherMatmul, kNameAllGatherMatmul, ADPT_DESC(AllGatherMatmul))
}  // namespace mindspore::transform
