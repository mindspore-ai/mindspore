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
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;

  // Case 1: 2D inputs with 1D target, reduction=MEAN -> scalar output
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{10, 8}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{10, 8}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{10}, kNumberTypeInt64},
                    // margin (float scalar)
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    // reduction = MEAN
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::MEAN))}})
    .FeedExpectedOutput({{}}, {kNumberTypeFloat32});

  // Case 2: 2D inputs with 1D target, reduction=NONE -> shape equals target (N)
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{6, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{6, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{6}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.5)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::NONE))}})
    .FeedExpectedOutput({{6}}, {kNumberTypeFloat32});

  // Case 3: 1D inputs with scalar target, reduction=SUM -> scalar output
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{8}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(-0.5)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::REDUCTION_SUM))}})
    .FeedExpectedOutput({{}}, {kNumberTypeFloat32});

  // Case 4: Broadcastable inputs, reduction=NONE -> output matches target shape
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{6, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{6, 1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{6}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::NONE))}})
    .FeedExpectedOutput({{6}}, {kNumberTypeFloat32});

  // Case 5: Invalid - input rank mismatch with target rank (inputs 3D, target 1D) -> throw
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::MEAN))}})
    .CaseShouldThrow();

  // Case 6: Invalid - target rank not in {0D, 1D} -> throw
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{5, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{5, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{5, 1}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::MEAN))}})
    .CaseShouldThrow();

  // Case 7: Dynamic shape input with (5, -1), (5, -1), (5) -> scalar output
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{5, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{5, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{5}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::MEAN))}})
    .FeedExpectedOutput({{}}, {kNumberTypeFloat32});

  // Case 8: Dynamic shape input with (5, -1), (-1, 5), (5), reduction=MEAN -> scalar output
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{5, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1, 5}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{5}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::MEAN))}})
    .FeedExpectedOutput({{}}, {kNumberTypeFloat32});

  // Case 9: Dynamic shape input with (-1, -1), (-1, -1), (5), reduction=NONE -> (5)
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{5}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::NONE))}})
    .FeedExpectedOutput({{5}}, {kNumberTypeFloat32});

  // Case 10: Dynamic shape input with (-1), (-1), scalar target, reduction=NONE -> scalar output
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::NONE))}})
    .FeedExpectedOutput({{}}, {kNumberTypeFloat32});

  // Case 11: Dynamic shape input with (-2), (-1, -1), (5), reduction=NONE -> (-2)
  // CosineEmbeddingLoss is composed by multiple little operators. Its output shape depends on mul and
  // select operators，which both support broadcastable input, making output shape can not be decided by 
  // the shape of "target". The same below.
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{5}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::NONE))}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  // Case 12: Dynamic shape input with (-2), (-1, -1), (-1), reduction=NONE -> (-2)
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::NONE))}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  // Case 13: Dynamic shape input with (-1, -1), (-2), (-2), reduction=SUM -> scalar output
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::REDUCTION_SUM))}})
    .FeedExpectedOutput({{}}, {kNumberTypeFloat32});

  // Case 14: Dynamic shape input with (-2), (-2), (-1), reduction=NONE -> (-2)
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::NONE))}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  // Case 15: Dynamic shape input with (-2), (-2), scalar target, reduction=MEAN -> scalar output
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::MEAN))}})
    .FeedExpectedOutput({{}}, {kNumberTypeFloat32});

  // Case 16: Dynamic shape input with (-2), (-2), (-2), reduction=NONE -> (-2)
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::NONE))}})
    .FeedExpectedOutput({{ShapeVector{-2}}}, {kNumberTypeFloat32});

  // Case 17: Dynamic shape input with (-2), (-2), (-2), reduction=SUM -> scalar output
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::REDUCTION_SUM))}})
    .FeedExpectedOutput({{}}, {kNumberTypeFloat32});

  // Case 18: Dynamic shape input with (-1, -1), (-1, -1), (-2), reduction=NONE -> (-2)
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::NONE))}})
    .FeedExpectedOutput({{ShapeVector{-2}}}, {kNumberTypeFloat32});

  // Case 19: Dynamic shape input with (-2), (5, -1), (-2), reduction=NONE -> (-2)
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{5, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::NONE))}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});

  // Case 20: Invalid - Dynamic shape input with (-2), (5, -1), (-1, -1), reduction=NONE -> throw
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{5, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1, -1}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::NONE))}})
    .CaseShouldThrow();

  // Case 21: Invalid - Dynamic shape input with (-1), (-1), (-1), reduction=NONE -> throw
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<double>(0.0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   CreatePyInt(static_cast<int64_t>(Reduction::NONE))}})
    .CaseShouldThrow();

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(CosineEmbeddingLoss, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops


