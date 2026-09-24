/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

namespace mindspore {
namespace ops {
namespace {
std::vector<GeneralInferParam> AbsInferParams() {
  auto params = single_input_eltwise_op_params();
  GeneralInferParamGenerator generator;
  generator.FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeComplex64}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});
  generator.FeedInputArgs({InferInfoParam{ShapeVector{2, -1}, kNumberTypeComplex128}})
    .FeedExpectedOutput({{2, -1}}, {kNumberTypeFloat64});
  auto complex_params = generator.Generate();
  params.insert(params.end(), complex_params.begin(), complex_params.end());
  return params;
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Abs, GeneralInferTest, testing::ValuesIn(AbsInferParams()));
}  // namespace ops
}  // namespace mindspore
