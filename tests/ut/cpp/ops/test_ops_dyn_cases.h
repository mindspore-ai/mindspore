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
namespace mindspore::ops {
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_Int8;
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_Int16;
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_Int32;
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_Int64;
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_UInt8;
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_UInt16;
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_UInt32;
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_UInt64;
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_Float16;
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_Float32;
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_Float64;
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_Complex64;
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_Complex128;
extern testing::internal::ValueArray<EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams, EltwiseOpParams, EltwiseOpParams, EltwiseOpParams,
                                     EltwiseOpParams>
  EltwiseDynTestCase_Bool;
}  // namespace mindspore::ops
#endif  // MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_DYN_CASES_H_
