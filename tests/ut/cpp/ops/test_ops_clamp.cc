/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_clamp_tensor_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4, 5}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3, 4, 1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3, 1, 5}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{3, 4, 5}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4, 5}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3, 4, 1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3, 1, 5}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{3, 4, 5}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4, 5}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3, 4, 1}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3, 1, 5}, kNumberTypeInt64}})
    .FeedExpectedOutput({{3, 4, 5}}, {kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4, 5}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{3, 4, 1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3, 1, 5}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{3, 4, 5}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{1}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{1}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  // Exception: min and max are both None.
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4, 5}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .CaseShouldThrow();
  // Exception: bool dtype is not allowed.
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4, 5}, kNumberTypeBool},
                    InferInfoParam{ShapeVector{3, 4, 1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3, 1, 5}, kNumberTypeFloat32}})
    .CaseShouldThrow();
  // Exception: min shape can not broadcast to input shape.
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{1}, kNumberTypeFloat32}})
    .CaseShouldThrow();
  return generator.Generate();
}

std::vector<GeneralInferParam> prepare_clamp_scalar_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4, 5}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.0f)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)}})
    .FeedExpectedOutput({{3, 4, 5}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4, 5}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.0f)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)}})
    .FeedExpectedOutput({{3, 4, 5}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4, 5}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{3, 4, 5}}, {kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4, 5}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.0f)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)}})
    .FeedExpectedOutput({{3, 4, 5}}, {kNumberTypeFloat16});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.0f)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)}})
    .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.0f)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  // Exception: min and max are both None.
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4, 5}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .CaseShouldThrow();
  // Exception: bool dtype is not allowed.
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4, 5}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)}})
    .CaseShouldThrow();
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(ClampTensor, GeneralInferTest, testing::ValuesIn(prepare_clamp_tensor_params()));
INSTANTIATE_TEST_CASE_P(ClampScalar, GeneralInferTest, testing::ValuesIn(prepare_clamp_scalar_params()));
}  // namespace mindspore::ops
