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

#include "ops/utils/general_infer_utils.h"
#include "ops/test_value_utils.h"
#include "op_def/op_enum.h"

namespace mindspore::ops {
namespace {
constexpr int64_t kLayoutBSND = static_cast<int64_t>(FASInputLayoutMode::BSND);
constexpr int64_t kLayoutTND = static_cast<int64_t>(FASInputLayoutMode::TND);

std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;

  // Case 1: BSND layout - query 4D, key 4D, dy 4D, all fixed shapes
  // Output: dq_shape = query_shape, dk_shape = key_shape, dw_shape = dy_shape
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 8, 16, 64}, kNumberTypeFloat16},     // query (B, S, N, D)
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat16},    // key (B, S, N, D)
                    InferInfoParam{ShapeVector{2, 8, 16, 2048}, kNumberTypeFloat16},   // dy
                    InferInfoParam{ShapeVector{2, 8, 16, 2048}, kNumberTypeInt32},     // sparse_indices
                    InferInfoParam{ShapeVector{2, 8, 16, 2048}, kNumberTypeFloat16},   // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},    // actual_seq_lengths_query
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},    // actual_seq_lengths_key
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},  // layout_query
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},     // sparse_mode
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},  // pre_tokens
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},})  // next_tokens
    .FeedExpectedOutput({{2, 8, 16, 64}, {2, 32, 16, 64}, {2, 8, 16, 2048}},
                        {kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});

  // Case 2: TND layout - query 3D, key 3D, dy 3D, all fixed shapes
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{4, 16, 64}, kNumberTypeBFloat16},       // query (T, N, D)
                    InferInfoParam{ShapeVector{4, 32, 64}, kNumberTypeBFloat16},       // key (T, N, D)
                    InferInfoParam{ShapeVector{4, 32, 1024}, kNumberTypeBFloat16},     // dy
                    InferInfoParam{ShapeVector{4, 32, 1024}, kNumberTypeInt32},        // sparse_indices
                    InferInfoParam{ShapeVector{4, 32, 1024}, kNumberTypeBFloat16},     // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},})
    .FeedExpectedOutput({{4, 16, 64}, {4, 32, 64}, {4, 32, 1024}},
                        {kNumberTypeBFloat16, kNumberTypeBFloat16, kNumberTypeBFloat16});

  // Case 3: Query dynamic rank
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},               // query dynamic rank
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat32},    // key 4D
                    InferInfoParam{ShapeVector{2, 8, 16, 512}, kNumberTypeFloat32},    // dy 4D
                    InferInfoParam{ShapeVector{2, 8, 16, 512}, kNumberTypeInt32},      // sparse_indices
                    InferInfoParam{ShapeVector{2, 8, 16, 512}, kNumberTypeFloat32},    // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},})
    .FeedExpectedOutput({{-2}, {2, 32, 16, 64}, {2, 8, 16, 512}},
                        {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});

  // Case 4: Key dynamic rank
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 8, 16, 64}, kNumberTypeFloat16},     // query 4D
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},               // key dynamic rank
                    InferInfoParam{ShapeVector{2, 8, 16, 1024}, kNumberTypeFloat16},   // dy 4D
                    InferInfoParam{ShapeVector{2, 8, 16, 1024}, kNumberTypeInt32},     // sparse_indices
                    InferInfoParam{ShapeVector{2, 8, 16, 1024}, kNumberTypeFloat16},   // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},})
    .FeedExpectedOutput({{2, 8, 16, 64}, {-2}, {2, 8, 16, 1024}},
                        {kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});

  // Case 5: Dy dynamic rank
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 8, 16, 64}, kNumberTypeBFloat16},    // query 4D
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeBFloat16},   // key 4D
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},              // dy dynamic rank
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt32},                 // sparse_indices
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},              // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},})
    .FeedExpectedOutput({{2, 8, 16, 64}, {2, 32, 16, 64}, {-2}},
                        {kNumberTypeBFloat16, kNumberTypeBFloat16, kNumberTypeBFloat16});

  // Case 6: All inputs have dynamic rank
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},               // query dynamic rank
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},               // key dynamic rank
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},               // dy dynamic rank
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt32},                 // sparse_indices
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},               // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},})
    .FeedExpectedOutput({{-2}, {-2}, {-2}},
                        {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});

  // Case 7: Partial dynamic dims on query and key
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, 16, 64}, kNumberTypeFloat16},   // query partially dynamic
                    InferInfoParam{ShapeVector{2, -1, 16, 64}, kNumberTypeFloat16},    // key partially dynamic
                    InferInfoParam{ShapeVector{2, 8, 16, 2048}, kNumberTypeFloat16},   // dy 4D fixed
                    InferInfoParam{ShapeVector{2, 8, 16, 2048}, kNumberTypeInt32},     // sparse_indices
                    InferInfoParam{ShapeVector{2, 8, 16, 2048}, kNumberTypeFloat16},   // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},})
    .FeedExpectedOutput({{-1, -1, 16, 64}, {2, -1, 16, 64}, {2, 8, 16, 2048}},
                        {kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});

  // Case 8: TND layout with partial dynamic dims
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 16, 64}, kNumberTypeBFloat16},      // query 3D partially dynamic
                    InferInfoParam{ShapeVector{8, -1, 64}, kNumberTypeBFloat16},       // key 3D partially dynamic
                    InferInfoParam{ShapeVector{8, 32, -1}, kNumberTypeBFloat16},       // dy 3D partially dynamic
                    InferInfoParam{ShapeVector{8, 32, -1}, kNumberTypeInt32},          // sparse_indices
                    InferInfoParam{ShapeVector{8, 32, -1}, kNumberTypeBFloat16},       // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},})
    .FeedExpectedOutput({{-1, 16, 64}, {8, -1, 64}, {8, 32, -1}},
                        {kNumberTypeBFloat16, kNumberTypeBFloat16, kNumberTypeBFloat16});

  // Case 9: With actual_seq_lengths tensors provided
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 8, 16, 64}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2, 8, 16, 1024}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2, 8, 16, 1024}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{2, 8, 16, 1024}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2}, kNumberTypeInt64},                  // actual_seq_lengths_query
                    InferInfoParam{ShapeVector{2}, kNumberTypeInt64},                  // actual_seq_lengths_key
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},})
    .FeedExpectedOutput({{2, 8, 16, 64}, {2, 32, 16, 64}, {2, 8, 16, 1024}},
                        {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(LightningIndexerGrad, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
