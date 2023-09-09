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

namespace mindspore::transform {
// KVCacheMgr
INPUT_MAP(KVCacheMgr) = {{1, INPUT_DESC(past)}, {2, INPUT_DESC(cur)}, {3, INPUT_DESC(index)}};
ATTR_MAP(KVCacheMgr) = EMPTY_ATTR_MAP;
OUTPUT_MAP(KVCacheMgr) = {{0, OUTPUT_DESC(past)}};
REG_ADPT_DESC(KVCacheMgr, "KVCacheMgr", ADPT_DESC(KVCacheMgr))

// FlashAttention
INPUT_MAP(FlashAttention) = {
  {1, INPUT_DESC(q)}, {2, INPUT_DESC(k)}, {3, INPUT_DESC(v)}, {4, INPUT_DESC(attention_mask)}};
ATTR_MAP(FlashAttention) = EMPTY_ATTR_MAP;
OUTPUT_MAP(FlashAttention) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FlashAttention, ops::kNameFlashAttention, ADPT_DESC(FlashAttention))

// MoeFFN
INPUT_MAP(MoeFFN) = {{1, INPUT_DESC(x)},          {2, INPUT_DESC(expert_tokens)}, {3, INPUT_DESC(weight1)},
                     {4, INPUT_DESC(bias1)},      {5, INPUT_DESC(weight2)},       {6, INPUT_DESC(bias2)},
                     {7, INPUT_DESC(scale)},      {8, INPUT_DESC(offset)},        {9, INPUT_DESC(deq_scale1)},
                     {10, INPUT_DESC(deq_scale2)}};
ATTR_MAP(MoeFFN) = {{"activation", ATTR_DESC(activation, AnyTraits<string>())}};
OUTPUT_MAP(MoeFFN) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MoeFFN, kNameMoeFFN, ADPT_DESC(MoeFFN))
}  // namespace mindspore::transform
