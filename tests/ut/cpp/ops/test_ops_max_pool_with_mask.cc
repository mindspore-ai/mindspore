/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
  // 4D (N, C, H, W) with all attrs statically known.
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{2, 3, 8, 8}, kNumberTypeFloat32},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
       InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(static_cast<int64_t>(kNumberTypeInt64))}})
    .FeedExpectedOutput({{2, 3, 4, 4}, {2, 3, 4, 64}}, {kNumberTypeFloat32, kNumberTypeInt8});

  // 3D (C, H, W) with all attrs statically known -> static branch, rank 3.
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{3, 8, 8}, kNumberTypeFloat32},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
       InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(static_cast<int64_t>(kNumberTypeInt64))}})
    .FeedExpectedOutput({{3, 4, 4}, {3, 4, 64}}, {kNumberTypeFloat32, kNumberTypeInt8});

  // 3D (C, H, W) with kernel_size not known at infer time -> fallback branch.
  // The static branch returns rank 3 for a 3D input, so the fallback must also keep rank 3.
  generator
    .FeedInputArgs(
      {InferInfoParam{ShapeVector{3, 8, 8}, kNumberTypeFloat32},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny},
       InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
       InferInfoParam{ShapeArray{{}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64},
                      ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
       InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
       InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(static_cast<int64_t>(kNumberTypeInt64))}})
    .FeedExpectedOutput({{3, -1, -1}, {3, -1, -1}}, {kNumberTypeFloat32, kNumberTypeInt8});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(MaxPoolWithMask, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
