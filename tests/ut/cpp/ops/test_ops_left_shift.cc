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
std::vector<GeneralInferParam> PrepareParams() {
  GeneralInferParamGenerator generator;

  // static shape broadcast cases
  generator.FeedInputArgs({InferInfoParam{ShapeVector{1, 3}, kNumberTypeInt8},
                           InferInfoParam{ShapeVector{2, 1}, kNumberTypeInt8}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt8});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeInt16},
                           InferInfoParam{ShapeVector{4}, kNumberTypeInt16}})
    .FeedExpectedOutput({{2, 3, 4}}, {kNumberTypeInt16});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{3, 1, 5}, kNumberTypeInt32},
                           InferInfoParam{ShapeVector{1, 4, 1}, kNumberTypeInt32}})
    .FeedExpectedOutput({{3, 4, 5}}, {kNumberTypeInt32});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{2, 1, 4, 5, 6, 9}, kNumberTypeUInt64},
                           InferInfoParam{ShapeVector{9}, kNumberTypeUInt64}})
    .FeedExpectedOutput({{2, 1, 4, 5, 6, 9}}, {kNumberTypeUInt64});

  // dynamic shape broadcast cases
  generator.FeedInputArgs({InferInfoParam{ShapeVector{-1, 3}, kNumberTypeInt64},
                           InferInfoParam{ShapeVector{-1, 1}, kNumberTypeInt64}})
    .FeedExpectedOutput({{-1, 3}}, {kNumberTypeInt64});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{-1, 1, 3}, kNumberTypeUInt8},
                           InferInfoParam{ShapeVector{1, -1, 3}, kNumberTypeUInt8}})
    .FeedExpectedOutput({{-1, -1, 3}}, {kNumberTypeUInt8});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{2, -1, 5}, kNumberTypeUInt16},
                           InferInfoParam{ShapeVector{1, -1, 1}, kNumberTypeUInt16}})
    .FeedExpectedOutput({{2, -1, 5}}, {kNumberTypeUInt16});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4, -1}, kNumberTypeUInt32},
                           InferInfoParam{ShapeVector{-1, -1, 4, 5}, kNumberTypeUInt32}})
    .FeedExpectedOutput({{2, 3, 4, 5}}, {kNumberTypeUInt32});

  // dynamic rank cases
  generator.FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeInt16},
                           InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt16}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeInt16});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeUInt32},
                           InferInfoParam{ShapeVector{-2}, kNumberTypeUInt32}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeUInt32});

  generator.FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeUInt64},
                           InferInfoParam{ShapeVector{-2}, kNumberTypeUInt64}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeUInt64});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(LeftShift, GeneralInferTest, testing::ValuesIn(PrepareParams()));
}  // namespace mindspore::ops
