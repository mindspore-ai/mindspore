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
#include "ops/erfinv.h"
#include "ops/test_ops_dyn_cases.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
class TestErfinv : public TestOps,
                   public testing::WithParamInterface<std::tuple<EltwiseOpShapeParams, EltwiseOpTypeParams>> {};

TEST_P(TestErfinv, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  auto expect = std::make_shared<abstract::AbstractTensor>(dtype_param.out_type, shape_param.out_shape);
  ASSERT_NE(x, nullptr);
  auto prim = std::make_shared<Primitive>(kNameErfinv);
  auto out_abstract = opt::CppInferShapeAndType(prim, {x});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

auto ErfinvOpTypeCases = testing::ValuesIn({
  EltwiseOpTypeParams{kFloat16, kFloat16},
  EltwiseOpTypeParams{kFloat32, kFloat32},
  EltwiseOpTypeParams{kFloat64, kFloat64},
});

INSTANTIATE_TEST_CASE_P(TestErfinv, TestErfinv, testing::Combine(EltwiseDynShapeTestCases, ErfinvOpTypeCases));
}  // namespace ops
}  // namespace mindspore
