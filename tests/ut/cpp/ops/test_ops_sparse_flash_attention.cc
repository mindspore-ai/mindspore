/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "mindspore/ops/op_def/op_name.h"
#include "infer/ops_func_impl/sparse_flash_attention.h"
#include "ops/test_value_utils.h"
#include "ops/utils/general_infer_utils.h"
#include "ops/utils/general_infer_param.h"
#include "ops/op_def.h"
#include "ir/value.h"

namespace mindspore {
namespace ops {

// Forward declarations
static std::vector<GeneralInferParam> prepare_params_sparse_flash_attention();

constexpr ShapeValueDType kShapeRankAny = mindspore::abstract::Shape::kShapeRankAny;
constexpr ShapeValueDType kShapeDimAny = mindspore::abstract::Shape::kShapeDimAny;

static std::vector<GeneralInferParam> prepare_params_sparse_flash_attention() {
  GeneralInferParamGenerator gen;

  // Case 0: BSND layout, fp16
  // query [B, S1, N1, D], key [B, S2, N2, D], value [B, S2, N2, D], sparse_indices [B, S1, N2, sparse_count]
  // Output: attention_out [B, S1, N1, D], softmax_max [B, N2, S1, N1/N2], softmax_sum [B, N2, S1, N1/N2]
  {
    const int64_t B = 4, S1 = 1, S2 = 8192, N1 = 128, N2 = 1, D = 512, D_rope = 64, sparse_count = 2048;
    gen
      .FeedInputArgs({
        InferInfoParam{ShapeVector{B, S1, N1, D}, kNumberTypeFloat16},                   // query
        InferInfoParam{ShapeVector{B, S2, N2, D}, kNumberTypeFloat16},                   // key
        InferInfoParam{ShapeVector{B, S2, N2, D}, kNumberTypeFloat16},                   // value
        InferInfoParam{ShapeVector{B, S1, N2, sparse_count}, kNumberTypeInt32},          // sparse_indices
        InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.044f)},  // scale_value
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},                  // block_table=None
        InferInfoParam{ShapeVector{B}, kNumberTypeInt32},                                // actual_seq_lengths_query [B]
        InferInfoParam{ShapeVector{B}, kNumberTypeInt32},                                // actual_seq_lengths_kv [B]
        InferInfoParam{ShapeVector{B, S1, N1, D_rope}, kNumberTypeFloat16},              // query_rope
        InferInfoParam{ShapeVector{B, S2, N2, D_rope}, kNumberTypeFloat16},              // key_rope
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},       // sparse_block_size=1
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},       // layout_query=BSND(3)
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},       // layout_kv=BSND(3)
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},       // sparse_mode=3
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},  // pre_tokens
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},  // next_tokens
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},  // attention_mode=2
        InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},  // return_softmax_lse=False
      })
      .FeedExpectedOutput({ShapeVector{B, S1, N1, D}, ShapeVector{B, N2, S1, N1 / N2}, ShapeVector{B, N2, S1, N1 / N2}},
                          {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  }

  // Case 1: BSND layout, bfloat16
  {
    const int64_t B = 2, S1 = 1, S2 = 4096, N1 = 64, N2 = 1, D = 512, D_rope = 64, sparse_count = 1024;
    gen
      .FeedInputArgs({
        InferInfoParam{ShapeVector{B, S1, N1, D}, kNumberTypeBFloat16},
        InferInfoParam{ShapeVector{B, S2, N2, D}, kNumberTypeBFloat16},
        InferInfoParam{ShapeVector{B, S2, N2, D}, kNumberTypeBFloat16},
        InferInfoParam{ShapeVector{B, S1, N2, sparse_count}, kNumberTypeInt32},
        InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.044f)},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},       // block_table=None
        InferInfoParam{ShapeVector{B}, kNumberTypeInt32},                     // actual_seq_lengths_query [B]
        InferInfoParam{ShapeVector{B}, kNumberTypeInt32},                     // actual_seq_lengths_kv [B]
        InferInfoParam{ShapeVector{B, S1, N1, D_rope}, kNumberTypeBFloat16},  // query_rope
        InferInfoParam{ShapeVector{B, S2, N2, D_rope}, kNumberTypeBFloat16},  // key_rope
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},  // attention_mode=2
        InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
      })
      .FeedExpectedOutput({ShapeVector{B, S1, N1, D}, ShapeVector{B, N2, S1, N1 / N2}, ShapeVector{B, N2, S1, N1 / N2}},
                          {kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  }

  // Case 2: BSND layout with RoPE tensors
  {
    const int64_t B = 4, S1 = 1, S2 = 8192, N1 = 128, N2 = 1, D = 512, D_rope = 64, sparse_count = 2048;
    gen
      .FeedInputArgs({
        InferInfoParam{ShapeVector{B, S1, N1, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{B, S2, N2, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{B, S2, N2, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{B, S1, N2, sparse_count}, kNumberTypeInt32},
        InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.044f)},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},      // block_table=None
        InferInfoParam{ShapeVector{B}, kNumberTypeInt32},                    // actual_seq_lengths_query [B]
        InferInfoParam{ShapeVector{B}, kNumberTypeInt32},                    // actual_seq_lengths_kv [B]
        InferInfoParam{ShapeVector{B, S1, N1, D_rope}, kNumberTypeFloat16},  // query_rope
        InferInfoParam{ShapeVector{B, S2, N2, D_rope}, kNumberTypeFloat16},  // key_rope
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},  // attention_mode=2
        InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},   // return_softmax_lse=True
      })
      .FeedExpectedOutput({ShapeVector{B, S1, N1, D}, ShapeVector{B, N2, S1, N1 / N2}, ShapeVector{B, N2, S1, N1 / N2}},
                          {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  }

  // Case 3: TND layout, fp16
  // query [T1, N1, D], key [T2, N2, D], value [T2, N2, D]
  // Output: attention_out [T1, N1, D], softmax_max [N2, T1, N1/N2], softmax_sum [N2, T1, N1/N2]
  {
    const int64_t T1 = 4, T2 = 8192, N1 = 128, N2 = 1, D = 512, D_rope = 64, sparse_count = 2048;
    gen
      .FeedInputArgs({
        InferInfoParam{ShapeVector{T1, N1, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{T2, N2, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{T2, N2, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{T1, N2, sparse_count}, kNumberTypeInt32},  // sparse_indices for TND
        InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.044f)},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // block_table=None
        InferInfoParam{ShapeVector{}, kMetaTypeNone,
                       mindspore::kNone},  // actual_seq_lengths_query=None (TND doesn't use batch)
        InferInfoParam{ShapeVector{}, kMetaTypeNone,
                       mindspore::kNone},  // actual_seq_lengths_kv=None (TND doesn't use batch)
        InferInfoParam{ShapeVector{T1, N1, D_rope}, kNumberTypeFloat16},  // query_rope
        InferInfoParam{ShapeVector{T2, N2, D_rope}, kNumberTypeFloat16},  // key_rope
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},  // layout_query=TND(4)
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},  // layout_kv=TND(4)
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},  // attention_mode=2
        InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
      })
      .FeedExpectedOutput({ShapeVector{T1, N1, D}, ShapeVector{N2, T1, N1 / N2}, ShapeVector{N2, T1, N1 / N2}},
                          {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  }

  // Case 4: BSND with dynamic query shape
  {
    const int64_t D_rope = 64;
    gen
      .FeedInputArgs({
        InferInfoParam{ShapeVector{-1, -1, -1, -1}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{4, 8192, 1, 512}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{4, 8192, 1, 512}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{4, 1, 1, 2048}, kNumberTypeInt32},
        InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.044f)},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},       // block_table=None
        InferInfoParam{ShapeVector{4}, kNumberTypeInt32},                     // actual_seq_lengths_query [B]
        InferInfoParam{ShapeVector{4}, kNumberTypeInt32},                     // actual_seq_lengths_kv [B]
        InferInfoParam{ShapeVector{-1, -1, -1, D_rope}, kNumberTypeFloat16},  // query_rope
        InferInfoParam{ShapeVector{4, 8192, 1, D_rope}, kNumberTypeFloat16},  // key_rope
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},  // attention_mode=2
        InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
      })
      .FeedExpectedOutput({ShapeVector{-1, -1, -1, -1}, ShapeVector{-1, 1, -1, -1}, ShapeVector{-1, 1, -1, -1}},
                          {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  }

  // Case 5: Dynamic rank query
  {
    const int64_t D_rope = 64;
    gen
      .FeedInputArgs({
        InferInfoParam{ShapeVector{kShapeRankAny}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{4, 8192, 1, 512}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{4, 8192, 1, 512}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{4, 1, 1, 2048}, kNumberTypeInt32},
        InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.044f)},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},       // block_table=None
        InferInfoParam{ShapeVector{4}, kNumberTypeInt32},                     // actual_seq_lengths_query [B]
        InferInfoParam{ShapeVector{4}, kNumberTypeInt32},                     // actual_seq_lengths_kv [B]
        InferInfoParam{ShapeVector{kShapeRankAny}, kNumberTypeFloat16},       // query_rope
        InferInfoParam{ShapeVector{4, 8192, 1, D_rope}, kNumberTypeFloat16},  // key_rope
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},  // attention_mode=2
        InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
      })
      .FeedExpectedOutput({ShapeVector{kShapeRankAny}, ShapeVector{kShapeRankAny}, ShapeVector{kShapeRankAny}},
                          {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  }

  // Case 6: N1=32
  {
    const int64_t B = 2, S1 = 1, S2 = 4096, N1 = 32, N2 = 1, D = 512, D_rope = 64, sparse_count = 1024;
    gen
      .FeedInputArgs({
        InferInfoParam{ShapeVector{B, S1, N1, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{B, S2, N2, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{B, S2, N2, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{B, S1, N2, sparse_count}, kNumberTypeInt32},
        InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.044f)},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},      // block_table=None
        InferInfoParam{ShapeVector{B}, kNumberTypeInt32},                    // actual_seq_lengths_query [B]
        InferInfoParam{ShapeVector{B}, kNumberTypeInt32},                    // actual_seq_lengths_kv [B]
        InferInfoParam{ShapeVector{B, S1, N1, D_rope}, kNumberTypeFloat16},  // query_rope
        InferInfoParam{ShapeVector{B, S2, N2, D_rope}, kNumberTypeFloat16},  // key_rope
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},  // attention_mode=2
        InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
      })
      .FeedExpectedOutput({ShapeVector{B, S1, N1, D}, ShapeVector{B, N2, S1, N1 / N2}, ShapeVector{B, N2, S1, N1 / N2}},
                          {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  }

  // Case 7: sparse_mode=0
  {
    const int64_t B = 4, S1 = 1, S2 = 4096, N1 = 64, N2 = 1, D = 512, D_rope = 64, sparse_count = 1024;
    gen
      .FeedInputArgs({
        InferInfoParam{ShapeVector{B, S1, N1, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{B, S2, N2, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{B, S2, N2, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{B, S1, N2, sparse_count}, kNumberTypeInt32},
        InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.044f)},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},      // block_table=None
        InferInfoParam{ShapeVector{B}, kNumberTypeInt32},                    // actual_seq_lengths_query [B]
        InferInfoParam{ShapeVector{B}, kNumberTypeInt32},                    // actual_seq_lengths_kv [B]
        InferInfoParam{ShapeVector{B, S1, N1, D_rope}, kNumberTypeFloat16},  // query_rope
        InferInfoParam{ShapeVector{B, S2, N2, D_rope}, kNumberTypeFloat16},  // key_rope
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},  // sparse_mode=0
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},  // attention_mode=2
        InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
      })
      .FeedExpectedOutput({ShapeVector{B, S1, N1, D}, ShapeVector{B, N2, S1, N1 / N2}, ShapeVector{B, N2, S1, N1 / N2}},
                          {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  }

  // Case 8: sparse_block_size=64
  {
    const int64_t B = 4, S1 = 1, S2 = 8192, N1 = 128, N2 = 1, D = 512, D_rope = 64, sparse_count = 2048;
    gen
      .FeedInputArgs({
        InferInfoParam{ShapeVector{B, S1, N1, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{B, S2, N2, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{B, S2, N2, D}, kNumberTypeFloat16},
        InferInfoParam{ShapeVector{B, S1, N2, sparse_count}, kNumberTypeInt32},
        InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.044f)},
        InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},              // block_table=None
        InferInfoParam{ShapeVector{B}, kNumberTypeInt32},                            // actual_seq_lengths_query [B]
        InferInfoParam{ShapeVector{B}, kNumberTypeInt32},                            // actual_seq_lengths_kv [B]
        InferInfoParam{ShapeVector{B, S1, N1, D_rope}, kNumberTypeFloat16},          // query_rope
        InferInfoParam{ShapeVector{B, S2, N2, D_rope}, kNumberTypeFloat16},          // key_rope
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(64)},  // sparse_block_size=64
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
        InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},  // attention_mode=2
        InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
      })
      .FeedExpectedOutput({ShapeVector{B, S1, N1, D}, ShapeVector{B, N2, S1, N1 / N2}, ShapeVector{B, N2, S1, N1 / N2}},
                          {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  }

  return gen.Generate();
}
INSTANTIATE_TEST_CASE_P(SparseFlashAttention, GeneralInferTest,
                        testing::ValuesIn(prepare_params_sparse_flash_attention()));

}  // namespace ops
}  // namespace mindspore
