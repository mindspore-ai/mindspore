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

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(256)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeUInt8)}})
    .FeedExpectedOutput({{256}}, {kNumberTypeUInt8});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeInt64)}})
    .FeedExpectedOutput({{0}}, {kNumberTypeInt64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(128)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeInt8)}})
    .FeedExpectedOutput({{128}}, {kNumberTypeInt8});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(32678)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeInt16)}})
    .FeedExpectedOutput({{32678}}, {kNumberTypeInt16});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(32678)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeInt8)}})
    .CaseShouldThrow();  // overflow
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(RandpermExt, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
