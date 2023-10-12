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

#include "transform/graph_ir/op_declare/experiment_ops_declare.h"
#include <vector>
#include <string>
namespace mindspore::transform {
// PromptFlashAttention
INPUT_MAP(PromptFlashAttention) = {
  {1, INPUT_DESC(query)},
  {2, INPUT_DESC(key)},
  {3, INPUT_DESC(value)},
  {4, INPUT_DESC(atten_mask)},         // optional input
  {5, INPUT_DESC(padding_mask)},       // optional input
  {6, INPUT_DESC(actual_seq_lengths)}  // optional input
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
                                  {5, INPUT_DESC(padding_mask)},
                                  {6, INPUT_DESC(actual_seq_lengths)}};
DYN_INPUT_MAP(IncreFlashAttention) = {{2, DYN_INPUT_DESC(key)}, {3, DYN_INPUT_DESC(value)}};
ATTR_MAP(IncreFlashAttention) = {{"num_heads", ATTR_DESC(num_heads, AnyTraits<int64_t>())},
                                 {"input_layout", ATTR_DESC(input_layout, AnyTraits<std::string>())},
                                 {"scale_value", ATTR_DESC(scale_value, AnyTraits<float>())}};
OUTPUT_MAP(IncreFlashAttention) = {{0, OUTPUT_DESC(attention_out)}};
REG_ADPT_DESC(IncreFlashAttention, "IncreFlashAttention", ADPT_DESC(IncreFlashAttention))

// BlendFaceBgPartOne
INPUT_MAP(BlendFaceBgPartOne) = {{1, INPUT_DESC(face_img)}, {2, INPUT_DESC(face_rect)}, {3, INPUT_DESC(face_mask)},
                                 {4, INPUT_DESC(acc_face)}, {5, INPUT_DESC(acc_mask)},  {6, INPUT_DESC(max_mask)}};
ATTR_MAP(BlendFaceBgPartOne) = EMPTY_ATTR_MAP;
OUTPUT_MAP(BlendFaceBgPartOne) = {{0, OUTPUT_DESC(acc_face)}, {1, OUTPUT_DESC(acc_mask)}, {2, OUTPUT_DESC(max_mask)}};
REG_ADPT_DESC(BlendFaceBgPartOne, kNameBlendFaceBgPartOne, ADPT_DESC(BlendFaceBgPartOne))

// FlashAttentionScore
INPUT_MAP(FlashAttentionScore) = {
  {1, INPUT_DESC(query)},     {2, INPUT_DESC(key)},        {3, INPUT_DESC(value)},        {4, INPUT_DESC(atten_mask)},
  {5, INPUT_DESC(drop_mask)}, {6, INPUT_DESC(real_shift)}, {7, INPUT_DESC(padding_mask)},
};
ATTR_MAP(FlashAttentionScore) = {
  {"scale_value", ATTR_DESC(scale_value, AnyTraits<float>())},
  {"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())},
  {"pre_tokens", ATTR_DESC(pre_tockens, AnyTraits<int64_t>())},
  {"next_tokens", ATTR_DESC(next_tockens, AnyTraits<int64_t>())},
  {"head_num", ATTR_DESC(head_num, AnyTraits<int64_t>())},
  {"inner_precise", ATTR_DESC(inner_precise, AnyTraits<int64_t>())},
  {"input_layout", ATTR_DESC(input_layout, AnyTraits<std::string>())},
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
                                      {11, INPUT_DESC(padding_mask)}, {12, INPUT_DESC(softmax_in)}};
ATTR_MAP(FlashAttentionScoreGrad) = {
  {"scale_value", ATTR_DESC(scale_value, AnyTraits<float>())},
  {"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())},
  {"pre_tokens", ATTR_DESC(pre_tockens, AnyTraits<int64_t>())},
  {"next_tokens", ATTR_DESC(next_tockens, AnyTraits<int64_t>())},
  {"head_num", ATTR_DESC(head_num, AnyTraits<int64_t>())},
  {"inner_precise", ATTR_DESC(inner_precise, AnyTraits<int64_t>())},
  {"input_layout", ATTR_DESC(input_layout, AnyTraits<std::string>())},
};
OUTPUT_MAP(FlashAttentionScoreGrad) = {{0, OUTPUT_DESC(dq)}, {1, OUTPUT_DESC(dk)}, {2, OUTPUT_DESC(dv)}};
REG_ADPT_DESC(FlashAttentionScoreGrad, kNameFlashAttentionScoreGrad, ADPT_DESC(FlashAttentionScoreGrad))
}  // namespace mindspore::transform
