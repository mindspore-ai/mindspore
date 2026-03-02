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
  // Test case 1: Basic 2D input with float16, row_block_size=1, col_block_size=128
  // Input shape: [2, 256], block size: 1x128
  // Expected scale shape: [ceil(2/1), ceil(256/128)] = [2, 2]
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 256}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.0f)},
                    InferInfoParam{ShapeVector{}, kObjectTypeString, MakeValue("rint")},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeFloat8E5M2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(128)}})
    .FeedExpectedOutput({{2, 256}, {2, 2}}, {kNumberTypeFloat8E5M2, kNumberTypeFloat32});

  // Test case 2: Large input with float64
  // Input shape: [128, 512], block size: 8x256
  // Expected scale shape: [ceil(128/8), ceil(512/256)] = [16, 2]
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{128, 512}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.0f)},
                    InferInfoParam{ShapeVector{}, kObjectTypeString, MakeValue("rint")},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeFloat8E5M2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(256)}})
    .FeedExpectedOutput({{128, 512}, {16, 2}}, {kNumberTypeFloat8E5M2, kNumberTypeFloat32});

  // Test case 3: Non-uniform division
  // Input shape: [10, 100], block size: 3x30
  // Expected scale shape: [ceil(10/3), ceil(100/30)] = [4, 4]
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{10, 100}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.0f)},
                    InferInfoParam{ShapeVector{}, kObjectTypeString, MakeValue("rint")},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeFloat8E5M2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(30)}})
    .FeedExpectedOutput({{10, 100}, {4, 4}}, {kNumberTypeFloat8E5M2, kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(DynamicBlockQuant, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
