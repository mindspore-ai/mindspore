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
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/equal.h"
#include "ops/test_ops_dyn_cases.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
class TestEqual : public TestOps,
                  public testing::WithParamInterface<std::tuple<BroadcastOpShapeParams, BroadcastOpTypeParams>> {};

TEST_P(TestEqual, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto y = std::make_shared<abstract::AbstractTensor>(dtype_param.y_type, shape_param.y_shape);
  ASSERT_NE(y, nullptr);
  auto expect = std::make_shared<abstract::AbstractTensor>(dtype_param.out_type, shape_param.out_shape);
  ASSERT_NE(expect, nullptr);
  auto prim = std::make_shared<Primitive>(kNameEqual);
  auto out_abstract = opt::CppInferShapeAndType(prim, {x, y});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

auto EqualOpTypeCases = testing::ValuesIn({
  BroadcastOpTypeParams{kBool, kBool, kBool},
  BroadcastOpTypeParams{kInt8, kInt8, kBool},
  BroadcastOpTypeParams{kInt16, kInt16, kBool},
  BroadcastOpTypeParams{kInt32, kInt32, kBool},
  BroadcastOpTypeParams{kInt64, kInt64, kBool},
  BroadcastOpTypeParams{kUInt8, kUInt8, kBool},
  BroadcastOpTypeParams{kUInt16, kUInt16, kBool},
  BroadcastOpTypeParams{kUInt32, kUInt32, kBool},
  BroadcastOpTypeParams{kFloat16, kFloat16, kBool},
  BroadcastOpTypeParams{kFloat32, kFloat32, kBool},
  BroadcastOpTypeParams{kFloat64, kFloat64, kBool},
  BroadcastOpTypeParams{kComplex64, kComplex64, kBool},
  BroadcastOpTypeParams{kComplex128, kComplex128, kBool},
});

INSTANTIATE_TEST_CASE_P(TestEqual_1, TestEqual, testing::Combine(BroadcastOpShapeScalarTensorCases, EqualOpTypeCases));
INSTANTIATE_TEST_CASE_P(TestEqual_2, TestEqual, testing::Combine(BroadcastOpShapeTensorTensorCases, EqualOpTypeCases));
}  // namespace ops
}  // namespace mindspore
