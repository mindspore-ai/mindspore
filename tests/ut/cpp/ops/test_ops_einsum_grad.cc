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
#include "ops/grad/einsum_grad.h"
#include "ops/test_ops_dyn_cases.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
class TestEinsumGrad : public TestOps,
                       public testing::WithParamInterface<std::tuple<EltwiseOpShapeParams, EltwiseOpTypeParams>> {};

TEST_P(TestEinsumGrad, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto d_out = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{0});
  ASSERT_NE(d_out, nullptr);
  auto expect = std::make_shared<abstract::AbstractTensor>(dtype_param.out_type, shape_param.out_shape);
  ASSERT_NE(expect, nullptr);
  auto prim = std::make_shared<Primitive>(kNameEinsumGrad);
  auto out_abstract = opt::CppInferShapeAndType(prim, {x, d_out});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

auto EinsumGradOpTypeCases = testing::ValuesIn({
  EltwiseOpTypeParams{kBool, kBool},
  EltwiseOpTypeParams{kInt8, kInt8},
  EltwiseOpTypeParams{kInt16, kInt16},
  EltwiseOpTypeParams{kInt32, kInt32},
  EltwiseOpTypeParams{kInt64, kInt64},
  EltwiseOpTypeParams{kUInt8, kUInt8},
  EltwiseOpTypeParams{kUInt16, kUInt16},
  EltwiseOpTypeParams{kUInt32, kUInt32},
  EltwiseOpTypeParams{kUInt64, kUInt64},
  EltwiseOpTypeParams{kFloat16, kFloat16},
  EltwiseOpTypeParams{kFloat32, kFloat32},
  EltwiseOpTypeParams{kFloat64, kFloat64},
  EltwiseOpTypeParams{kComplex64, kComplex64},
  EltwiseOpTypeParams{kComplex128, kComplex128},
});

INSTANTIATE_TEST_CASE_P(TestEinsumGrad, TestEinsumGrad,
                        testing::Combine(EltwiseDynShapeTestCases, EinsumGradOpTypeCases));
}  // namespace ops
}  // namespace mindspore
