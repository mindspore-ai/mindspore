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

#include <vector>
#include <memory>

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_value_utils.h"
#include "ops/utils/general_infer_utils.h"
#include "ops/utils/general_infer_param.h"

namespace mindspore {
namespace ops {

static std::vector<GeneralInferParam> prepare_params_flash_attention_score_grad() {
  GeneralInferParamGenerator gen;

  // Common scalar attributes
  const int64_t layout_bsh = 0;  // BSH
  const int64_t sparse_default = 0;
  const int64_t head_num = 2;
  const float keep_prob = 1.0f;
  const float scale_value = 1.0f;
  const int64_t pre_tokens = 0;
  const int64_t next_tokens = 0;
  const int64_t inner_precise = 0;

  auto feed_case = [&](const ShapeVector &qkv_shape, TypeId dtype, const ShapeVector &pse_shape,
                       const ShapeArray &expect_shapes) {
    // Derive softmax shapes when static; otherwise pass None
    bool is_dynamic = IsDynamic(qkv_shape);
    ShapeVector softmax_shape{};
    if (!is_dynamic && qkv_shape.size() == 3) {
      const int64_t B = qkv_shape[0];
      const int64_t S = qkv_shape[1];
      softmax_shape = {B, head_num, S, 8};
    }

    gen.FeedInputArgs({
         // query, key, value, dy
         InferInfoParam{qkv_shape, dtype},
         InferInfoParam{qkv_shape, dtype},
         InferInfoParam{qkv_shape, dtype},
         InferInfoParam{qkv_shape, dtype},
         // pse_shift
         InferInfoParam{pse_shape, dtype},
         // drop_mask, padding_mask, attn_mask
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         // softmax_max, softmax_sum
         is_dynamic ? InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}
                    : InferInfoParam{softmax_shape, kNumberTypeFloat32},
         is_dynamic ? InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}
                    : InferInfoParam{softmax_shape, kNumberTypeFloat32},
         // softmax_in, attention_in
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         // prefix, actual_seq_qlen, actual_seq_kvlen
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         // head_num, keep_prob, scale_value, pre_tokens, next_tokens, inner_precise, input_layout, sparse_mode
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(head_num)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(keep_prob)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(scale_value)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(pre_tokens)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(next_tokens)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(inner_precise)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(layout_bsh)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(sparse_default)},
       })
       .FeedExpectedOutput(expect_shapes, {dtype, dtype, dtype, dtype});
  };

  // Case 0: dynamic rank (-2), fp16
  feed_case(ShapeVector{-2}, kNumberTypeFloat16, ShapeVector{-2},
            ShapeArray{ShapeVector{-2}, ShapeVector{-2}, ShapeVector{-2}, ShapeVector{-2}});

  // Case 1: dynamic rank (-2), bfloat16
  feed_case(ShapeVector{-2}, kNumberTypeBFloat16, ShapeVector{-2},
            ShapeArray{ShapeVector{-2}, ShapeVector{-2}, ShapeVector{-2}, ShapeVector{-2}});

  // Case 2: known rank with dynamic dims (-1), fp16
  feed_case(ShapeVector{-1, -1, -1}, kNumberTypeFloat16, ShapeVector{-1, -1, -1},
            ShapeArray{ShapeVector{-1, -1, -1}, ShapeVector{-1, -1, -1}, ShapeVector{-1, -1, -1},
                       ShapeVector{-1, -1, -1}});

  // Case 3: known rank with dynamic dims (-1), bfloat16
  feed_case(ShapeVector{-1, -1, -1}, kNumberTypeBFloat16, ShapeVector{-1, -1, -1},
            ShapeArray{ShapeVector{-1, -1, -1}, ShapeVector{-1, -1, -1}, ShapeVector{-1, -1, -1},
                       ShapeVector{-1, -1, -1}});

  // Case 4: static [4,6,8], fp16 (pse_shift shape must be [B, head_num, S, S])
  feed_case(ShapeVector{4, 6, 8}, kNumberTypeFloat16, ShapeVector{4, 2, 6, 6},
            ShapeArray{ShapeVector{4, 6, 8}, ShapeVector{4, 6, 8}, ShapeVector{4, 6, 8}, ShapeVector{4, 2, 6, 6}});

  // Case 5: static [4,6,8], bfloat16 (pse_shift shape must be [B, head_num, S, S])
  feed_case(ShapeVector{4, 6, 8}, kNumberTypeBFloat16, ShapeVector{4, 2, 6, 6},
            ShapeArray{ShapeVector{4, 6, 8}, ShapeVector{4, 6, 8}, ShapeVector{4, 6, 8}, ShapeVector{4, 2, 6, 6}});

  // Case 6: static [4,6,8], fp16, optional scalars unknown (mimic old UT skipping checks)
  gen.FeedInputArgs({
         InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
         // pse_shift
         InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeFloat16},
         // drop_mask, padding_mask, attn_mask
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         // softmax_max, softmax_sum (unset)
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         // softmax_in, attention_in
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         // prefix, actual_seq_qlen, actual_seq_kvlen
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         // head_num (unknown), keep_prob, scale_value, pre/next/inner, input_layout (unknown), sparse_mode (unknown)
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(kValueAny)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(kValueAny)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(kValueAny)},
       })
     .FeedExpectedOutput(
       {ShapeVector{4, 6, 8}, ShapeVector{4, 6, 8}, ShapeVector{4, 6, 8}, ShapeVector{4, 6, 8}},
       {kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});

  // Case 7: static [4,6,8], bfloat16, optional scalars unknown
  gen.FeedInputArgs({
         InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeBFloat16},
         InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeBFloat16},
         InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeBFloat16},
         InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeBFloat16},
         // pse_shift
         InferInfoParam{ShapeVector{4, 6, 8}, kNumberTypeBFloat16},
         // drop_mask, padding_mask, attn_mask
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         // softmax_max, softmax_sum (unset)
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         // softmax_in, attention_in
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         // prefix, actual_seq_qlen, actual_seq_kvlen
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         // head_num (unknown), keep_prob, scale_value, pre/next/inner, input_layout (unknown), sparse_mode (unknown)
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(kValueAny)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(kValueAny)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(kValueAny)},
       })
     .FeedExpectedOutput(
       {ShapeVector{4, 6, 8}, ShapeVector{4, 6, 8}, ShapeVector{4, 6, 8}, ShapeVector{4, 6, 8}},
       {kNumberTypeBFloat16, kNumberTypeBFloat16, kNumberTypeBFloat16, kNumberTypeBFloat16});

  return gen.Generate();
}

INSTANTIATE_TEST_CASE_P(FlashAttentionScoreGrad, GeneralInferTest,
                        testing::ValuesIn(prepare_params_flash_attention_score_grad()));

}  // namespace ops
}  // namespace mindspore
