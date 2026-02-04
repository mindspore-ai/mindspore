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
 * limitations under the License.a
 */

#include "ops/utils/general_infer_utils.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{24, 8, 96, 96}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{24, 8, 96, 96}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat64, CreateScalar<double>(1e-5f)},
      InferInfoParam{ShapeArray{{}, {}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
    })
    .FeedExpectedOutput({{24, 8, 96, 96}, {8}, {8}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{24, 8, 96, 96}, kNumberTypeFloat64},
      InferInfoParam{ShapeVector{24, 8, 96, 96}, kNumberTypeFloat64},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat64, CreateScalar<double>(1e-5f)},
      InferInfoParam{ShapeArray{{}, {}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
    })
    .FeedExpectedOutput({{24, 8, 96, 96}, {8}, {8}}, {kNumberTypeFloat64, kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{24, 8, 96, 96}, kNumberTypeFloat64},
      InferInfoParam{ShapeVector{24, 8, 96, 96}, kNumberTypeFloat64},
      InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat64},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat64},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat64},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat64},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat64, CreateScalar<double>(1e-5f)},
      InferInfoParam{ShapeArray{{}, {}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
    })
    .FeedExpectedOutput({{24, 8, 96, 96}, {8}, {8}}, {kNumberTypeFloat64, kNumberTypeFloat64, kNumberTypeFloat64});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, -1, -1, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, -1, -1, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat64, CreateScalar<double>(1e-5f)},
      InferInfoParam{ShapeArray{{}, {}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
    })
    .FeedExpectedOutput({{-1, -1, -1, -1}, {-1}, {-1}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat64, CreateScalar<double>(1e-5f)},
      InferInfoParam{ShapeArray{{}, {}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
    })
    .FeedExpectedOutput({{-2}, {-1}, {-1}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, 8, 64, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, 8, 64, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat64, CreateScalar<double>(1e-5f)},
      InferInfoParam{ShapeArray{{}, {}, {}}, TypeIdList{kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64},
                     ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
    })
    .FeedExpectedOutput({{-1, 8, 64, 64}, {8}, {8}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(BatchNormGradExt, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace ops
}  // namespace mindspore
