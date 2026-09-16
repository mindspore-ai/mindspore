/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(4), CreateScalar<int64_t>(4)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
       InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(static_cast<int64_t>(kNumberTypeInt64))}})
    .FeedExpectedOutput({{-2}, {-2}}, {kNumberTypeFloat16, kNumberTypeInt64});

  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{-1, -1, -1, -1}, kNumberTypeFloat16},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(4), CreateScalar<int64_t>(4)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
       InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(static_cast<int64_t>(kNumberTypeInt64))}})
    .FeedExpectedOutput({{-1, -1, -1, -1}, {-1, -1, -1, -1}}, {kNumberTypeFloat16, kNumberTypeInt64});

  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{1, 1, 8, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(4), CreateScalar<int64_t>(4)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
       InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(static_cast<int64_t>(kNumberTypeInt64))}})
    .FeedExpectedOutput({{1, 1, 2, 2}, {1, 1, 2, 2}}, {kNumberTypeFloat16, kNumberTypeInt64});

  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{1, 1, 8, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(4), CreateScalar<int64_t>(4)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
       InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(static_cast<int64_t>(kNumberTypeInt64))}})
    .FeedExpectedOutput({{1, 1, 3, 3}, {1, 1, 3, 3}}, {kNumberTypeFloat16, kNumberTypeInt64});

  // 3D (C, H, W) with kernel_size not known at infer time -> fallback branch, rank must stay 3.
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{3, 8, 8}, kNumberTypeFloat16},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
       InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(static_cast<int64_t>(kNumberTypeInt64))}})
    .FeedExpectedOutput({{3, -1, -1}, {3, -1, -1}}, {kNumberTypeFloat16, kNumberTypeInt64});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(MaxPoolWithIndices, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
