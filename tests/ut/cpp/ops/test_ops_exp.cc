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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "utils/ms_context.h"
#include "ops/test_ops.h"
#include "ops/exp.h"
#include "ops/test_ops_dyn_cases.h"

namespace mindspore {
namespace ops {
class TestExp : public TestOps, public testing::WithParamInterface<EltwiseOpParams> {};

TEST_P(TestExp, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto expect = std::make_shared<abstract::AbstractTensor>(param.out_type, param.out_shape);
  ASSERT_NE(x, nullptr);
  auto prim = std::make_shared<Primitive>(kNameExp);
  auto out_abstract = ExpInfer(nullptr, prim, {x});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(TestExp_int8, TestExp, EltwiseDynTestCase_Int8);
INSTANTIATE_TEST_CASE_P(TestExp_int16, TestExp, EltwiseDynTestCase_Int16);
INSTANTIATE_TEST_CASE_P(TestExp_int32, TestExp, EltwiseDynTestCase_Int32);
INSTANTIATE_TEST_CASE_P(TestExp_int64, TestExp, EltwiseDynTestCase_Int64);
INSTANTIATE_TEST_CASE_P(TestExp_uint8, TestExp, EltwiseDynTestCase_UInt8);
INSTANTIATE_TEST_CASE_P(TestExp_uint16, TestExp, EltwiseDynTestCase_UInt16);
INSTANTIATE_TEST_CASE_P(TestExp_uint32, TestExp, EltwiseDynTestCase_UInt32);
INSTANTIATE_TEST_CASE_P(TestExp_uint64, TestExp, EltwiseDynTestCase_UInt64);
INSTANTIATE_TEST_CASE_P(TestExp_fp16, TestExp, EltwiseDynTestCase_Float16);
INSTANTIATE_TEST_CASE_P(TestExp_fp32, TestExp, EltwiseDynTestCase_Float32);
INSTANTIATE_TEST_CASE_P(TestExp_fp64, TestExp, EltwiseDynTestCase_Float64);
INSTANTIATE_TEST_CASE_P(TestExp_complex64, TestExp, EltwiseDynTestCase_Complex64);
INSTANTIATE_TEST_CASE_P(TestExp_complex128, TestExp, EltwiseDynTestCase_Complex128);
INSTANTIATE_TEST_CASE_P(TestExp_bool, TestExp, EltwiseDynTestCase_Bool);
}  // namespace ops
}  // namespace mindspore
