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

  // Case 1: BSND layout - query 4D, key 4D, fixed sparse_count
  // Branch coverage: layout_query == BSND, query fixed rank 4D, key fixed rank 4D, sparse_count has value
  // Output shape: [query[0], query[1], key[2], sparse_count]
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 8, 16, 64}, kNumberTypeFloat16},    // query (B, S, N, D)
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat16},   // key (B, S, N, D)
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat16},   // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},   // actual_seq_lengths_query
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},   // actual_seq_lengths_key
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},   // block_table
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},  // layout_query
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},  // layout_key
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2048)},  // sparse_count
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},     // sparse_mode
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},  // pre_tokens
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},  // next_tokens
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})  // return_value
    .FeedExpectedOutput({{2, 8, 16, 2048}, {2, 8, 16, 2048}},
                        {kNumberTypeInt32, kNumberTypeFloat16});

  // Case 2: TND layout - query 3D, key 3D with TND layout_key, fixed sparse_count
  // Branch coverage: layout_query != BSND (TND), layout_key == TND, key_shape[1] used for output[1]
  // Output shape: [query[0], key[1], sparse_count]
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{4, 16, 64}, kNumberTypeBFloat16},      // query (T, N, D)
                    InferInfoParam{ShapeVector{4, 32, 64}, kNumberTypeBFloat16},      // key (T, N, D)
                    InferInfoParam{ShapeVector{4, 32, 64}, kNumberTypeBFloat16},      // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},   // actual_seq_lengths_query
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},   // actual_seq_lengths_key
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},   // block_table
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},   // layout_query
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},   // layout_key
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1024)},  // sparse_count
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},     // sparse_mode
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{4, 32, 1024}, {4, 32, 1024}},
                        {kNumberTypeInt32, kNumberTypeBFloat16});

  // Case 3: TND layout - query 3D, key 4D with BSND layout_key (non-TND)
  // Branch coverage: layout_query != BSND, layout_key != TND, key_shape[2] used for output[1]
  // Output shape: [query[0], key[2], sparse_count]
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{8, 16, 64}, kNumberTypeFloat32},       // query (T, N, D)
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat32},   // key (B, S, N, D)
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat32},   // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},   // layout_query = TND
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},  // layout_key = BSND
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(512)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{8, 16, 512}, {8, 16, 512}},
                        {kNumberTypeInt32, kNumberTypeFloat32});

  // Case 4: Query dynamic rank - BSND layout
  // Branch coverage: is_query_dynamic_rank = true, output first two dims are kShapeDimAny
  // Output shape: [-1, -1, key[2], sparse_count]
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},              // query dynamic rank
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat16},   // key 4D
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat16},   // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1024)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{-1, -1, 16, 1024}, {-1, -1, 16, 1024}},
                        {kNumberTypeInt32, kNumberTypeFloat16});

  // Case 5: Key dynamic rank - BSND layout
  // Branch coverage: is_key_dynamic_rank = true, output third dim is kShapeDimAny
  // Output shape: [query[0], query[1], -1, sparse_count]
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 8, 16, 64}, kNumberTypeBFloat16},   // query 4D
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},             // key dynamic rank
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},             // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2048)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{2, 8, -1, 2048}, {2, 8, -1, 2048}},
                        {kNumberTypeInt32, kNumberTypeBFloat16});

  // Case 6: Both query and key have dynamic rank - BSND layout
  // Branch coverage: both is_query_dynamic_rank and is_key_dynamic_rank are true
  // Output shape: [-1, -1, -1, sparse_count]
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(512)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{-1, -1, -1, 512}, {-1, -1, -1, 512}},
                        {kNumberTypeInt32, kNumberTypeFloat16});

  // Case 7: Query dynamic rank with TND layout
  // Branch coverage: is_query_dynamic_rank = true, layout_query = TND
  // Output shape: [-1, key_dim, sparse_count]
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},              // query dynamic rank
                    InferInfoParam{ShapeVector{8, 32, 64}, kNumberTypeFloat32},       // key 3D TND
                    InferInfoParam{ShapeVector{8, 32, 64}, kNumberTypeFloat32},       // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(256)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{-1, 32, 256}, {-1, 32, 256}},
                        {kNumberTypeInt32, kNumberTypeFloat32});

  // Case 8: Partial dynamic dims on query - BSND layout
  // Branch coverage: query has dynamic dims but not dynamic rank
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, 16, 64}, kNumberTypeFloat16},  // query partially dynamic
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat16},   // key 4D fixed
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat16},   // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1024)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{-1, -1, 16, 1024}, {-1, -1, 16, 1024}},
                        {kNumberTypeInt32, kNumberTypeFloat16});

  // Case 9: Partial dynamic dims on key - BSND layout
  // Branch coverage: key has dynamic dims but not dynamic rank
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 8, 16, 64}, kNumberTypeBFloat16},   // query 4D fixed
                    InferInfoParam{ShapeVector{2, -1, -1, 64}, kNumberTypeBFloat16},  // key partially dynamic
                    InferInfoParam{ShapeVector{2, -1, -1, 64}, kNumberTypeBFloat16},  // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2048)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{2, 8, -1, 2048}, {2, 8, -1, 2048}},
                        {kNumberTypeInt32, kNumberTypeBFloat16});

  // Case 10: Mixed scenario - query dynamic rank, key partially dynamic, TND layout
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},              // query dynamic rank
                    InferInfoParam{ShapeVector{4, -1, 64}, kNumberTypeFloat16},       // key partially dynamic 3D
                    InferInfoParam{ShapeVector{4, -1, 64}, kNumberTypeFloat16},       // weights
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1024)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{-1, -1, 1024}, {-1, -1, 1024}},
                        {kNumberTypeInt32, kNumberTypeFloat16});

  // Case 11: Full dynamic scenario - query dynamic rank, key dynamic rank, TND layout
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(256)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{-1, -1, 256}, {-1, -1, 256}},
                        {kNumberTypeInt32, kNumberTypeFloat32});

  // Case 12: layout_query is kValueAny - returns dynamic rank output
  // Branch coverage: !layout_query_opt.has_value(), returns {kShapeRankAny}
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 8, 16, 64}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny},        // layout_query is kValueAny
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2048)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{-2}, {-2}},
                        {kNumberTypeInt32, kNumberTypeFloat16});

  // Case 13: layout_key is kValueAny with TND layout - output[1] is kDimUnknown
  // Branch coverage: layout_query = TND, !layout_key_opt.has_value()
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{4, 16, 64}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{4, 32, 64}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{4, 32, 64}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny},        // layout_key is kValueAny
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(512)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{4, -1, 512}, {4, -1, 512}},
                        {kNumberTypeInt32, kNumberTypeBFloat16});

  // Case 14: sparse_count is kValueAny - last dim of output is kDimUnknown
  // Branch coverage: !sparse_count_opt.has_value()
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 8, 16, 64}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2, 32, 16, 64}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutBSND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny},        // sparse_count is kValueAny
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{2, 8, 16, -1}, {2, 8, 16, -1}},
                        {kNumberTypeInt32, kNumberTypeFloat32});

  // Case 15: sparse_count is kValueAny with TND layout
  // Branch coverage: layout_query = TND, !sparse_count_opt.has_value()
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{8, 16, 64}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{8, 32, 64}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{8, 32, 64}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kLayoutTND)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny},        // sparse_count is kValueAny
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(9223372036854775807LL)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},})
    .FeedExpectedOutput({{8, 32, -1}, {8, 32, -1}},
                        {kNumberTypeInt32, kNumberTypeFloat16});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(LightningIndexer, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
