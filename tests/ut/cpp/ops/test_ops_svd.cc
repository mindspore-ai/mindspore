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
#include "op_enum.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{3,}, {3, 3}, {4, 3}},
                        {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
  .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
  .FeedExpectedOutput({{2, 3}, {2, 3, 3}, {2, 4, 3}},
                      {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
  .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
  .FeedExpectedOutput({{2, 3}, {1}, {1}},
                      {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
  .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
  .FeedExpectedOutput({{2, 3}, {1}, {1}},
                      {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
  .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
  .FeedExpectedOutput({{2, 3}, {2, 3, 3}, {2, 4, 4}},
                      {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
  .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat32},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
  .FeedExpectedOutput({{-2}, {-2}, {-2}},
                      {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
  .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
  .FeedExpectedOutput({{-2}, {-2}, {-2}},
                      {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
  .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar(kValueAny)},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
  .FeedExpectedOutput({{2, 3}, {2, 3, -1}, {2, 4, -1}},
                      {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
  .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar(kValueAny)},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
  .FeedExpectedOutput({{2, 3}, {1}, {1}},
                      {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
  .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar(kValueAny)},
                  InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar(kValueAny)}})
  .FeedExpectedOutput({{2, 3}, {-2}, {-2}},
                      {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1,}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar(kValueAny)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar(kValueAny)}})
    .CaseShouldThrow();
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Svd, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
