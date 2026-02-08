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

namespace mindspore::ops {
namespace  {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  // static
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{5, 2, 1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2, 2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{2, 1}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{5, 2, 1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{0, 0}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{0, 0}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{0, 0, 0}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{0, 0}}, {kNumberTypeFloat32});
  // dynamic shape
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{5, -1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3, 2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{3, 4}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{5, -1, 4}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1, 1}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{-1, 3}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1, 3}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{5}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeFloat32});
  //dynamic rank
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2, 3, 2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{2, 3, 3, 4}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{-1, -1, 3, 4}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1, -1}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});
  return generator.Generate();
}
}  //namespace
INSTANTIATE_TEST_CASE_P(TensorScatterAdd, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore
