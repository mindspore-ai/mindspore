/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_DYN_CASES_H_
#define MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_DYN_CASES_H_

#include "common/common_test.h"
#include "gtest/internal/gtest-param-util.h"
#include "ops/test_ops.h"

namespace mindspore::ops {
extern testing::internal::ValueArray<
  EltwiseOpShapeParams, EltwiseOpShapeParams, EltwiseOpShapeParams, EltwiseOpShapeParams, EltwiseOpShapeParams,
  EltwiseOpShapeParams, EltwiseOpShapeParams, EltwiseOpShapeParams, EltwiseOpShapeParams, EltwiseOpShapeParams>
  EltwiseDynShapeTestCases;

extern testing::internal::ValueArray<EltwiseGradOpShapeParams, EltwiseGradOpShapeParams, EltwiseGradOpShapeParams,
                                     EltwiseGradOpShapeParams, EltwiseGradOpShapeParams, EltwiseGradOpShapeParams,
                                     EltwiseGradOpShapeParams, EltwiseGradOpShapeParams, EltwiseGradOpShapeParams,
                                     EltwiseGradOpShapeParams>
  EltwiseGradDynShapeTestCases;

extern testing::internal::ValueArray<BroadcastOpShapeParams, BroadcastOpShapeParams, BroadcastOpShapeParams,
                                     BroadcastOpShapeParams, BroadcastOpShapeParams, BroadcastOpShapeParams,
                                     BroadcastOpShapeParams, BroadcastOpShapeParams, BroadcastOpShapeParams,
                                     BroadcastOpShapeParams>
  BroadcastOpShapeScalarTensorCases;

extern testing::internal::ValueArray<BroadcastOpShapeParams, BroadcastOpShapeParams, BroadcastOpShapeParams,
                                     BroadcastOpShapeParams, BroadcastOpShapeParams, BroadcastOpShapeParams,
                                     BroadcastOpShapeParams, BroadcastOpShapeParams, BroadcastOpShapeParams,
                                     BroadcastOpShapeParams>
  BroadcastOpShapeTensorTensorCases;
}  // namespace mindspore::ops
#endif  // MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_DYN_CASES_H_
