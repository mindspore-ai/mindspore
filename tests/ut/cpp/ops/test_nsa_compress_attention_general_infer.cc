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
#include "ops/utils/general_infer_param.h"
#include "ops/op_def.h"
#include "ir/value.h"

namespace mindspore::ops {
namespace {
constexpr ShapeValueDType kShapeRankAny = mindspore::abstract::Shape::kShapeRankAny;
constexpr ShapeValueDType kShapeDimAny = mindspore::abstract::Shape::kShapeDimAny;

static std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator gen;
  // Case 0: valid basic case
  gen
    .FeedInputArgs({InferInfoParam{ShapeVector{128, 8, 64}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{128, 8, 64}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{128, 8, 64}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(32)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(64)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, nullptr},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, nullptr},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(128)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(128)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(128)}}})
    .FeedExpectedOutput(
      {ShapeVector{128, 8, 64}, ShapeVector{128, 8, 16}, ShapeVector{128, 8, 8}, ShapeVector{128, 8, 8}},
      {kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32});

  // Case 1: dynamic sequence length
  gen
    .FeedInputArgs({InferInfoParam{ShapeVector{kShapeDimAny, 8, 64}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{kShapeDimAny, 8, 64}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{kShapeDimAny, 8, 64}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(32)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(64)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, nullptr},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, nullptr},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(128)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(128)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(128)}}})
     .FeedExpectedOutput({ShapeVector{kShapeDimAny, 8, 64}, ShapeVector{kShapeDimAny, 8, 16},
                         ShapeVector{kShapeDimAny, 8, 8}, ShapeVector{kShapeDimAny, 8, 8}},
                        {kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32});

  // Case 2: dynamic head dimension
  gen
    .FeedInputArgs({InferInfoParam{ShapeVector{128, 8, kShapeDimAny}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{128, 8, kShapeDimAny}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{128, 8, kShapeDimAny}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(32)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(64)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, nullptr},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, nullptr},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(128)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(128)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(128)}}})
      .FeedExpectedOutput(
      {ShapeVector{128, 8, kShapeDimAny}, ShapeVector{128, 8, 16}, ShapeVector{128, 8, 8}, ShapeVector{128, 8, 8}},
      {kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32});

  // Case 3: with masks
  gen
    .FeedInputArgs({InferInfoParam{ShapeVector{64, 4, 32}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{64, 4, 32}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{64, 4, 32}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(32)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(64)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{64, 64}, kNumberTypeBool},
                    InferInfoParam{ShapeVector{64, 64}, kNumberTypeBool},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(64)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(64)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{64, 4, 32}, ShapeVector{64, 4, 16}, ShapeVector{64, 4, 8}, ShapeVector{64, 4, 8}},
                        {kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32});

  // Case 4: dynamic rank input
  gen
    .FeedInputArgs({InferInfoParam{ShapeVector{kShapeRankAny}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{kShapeRankAny}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{kShapeRankAny}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(32)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(64)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, nullptr},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, nullptr},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(128)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(128)}},
                    InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(128)}}})
    .FeedExpectedOutput(
      {ShapeVector{kShapeRankAny}, ShapeVector{kShapeRankAny}, ShapeVector{kShapeRankAny}, ShapeVector{kShapeRankAny}},
      {kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32});

  return gen.Generate();
}

}  // namespace

INSTANTIATE_TEST_CASE_P(NsaCompressAttention, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops