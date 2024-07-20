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
#include "ir/primitive.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/test_ops_dyn_cases.h"
#include "ops/ops_func_impl/hsigmoid_grad.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
class TestHSigmoidGrad
    : public TestOps,
      public testing::WithParamInterface<std::tuple<EltwiseGradOpShapeParams, EltwiseGradOpTypeParams>> {};

TEST_P(TestHSigmoidGrad, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  HSigmoidGradFuncImpl hsigmoid_grad_func_impl;
  auto prim = std::make_shared<Primitive>("HSigmoidGrad");

  auto grad = std::make_shared<abstract::AbstractTensor>(dtype_param.grad_type, shape_param.grad_shape);
  ASSERT_NE(grad, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_type = std::make_shared<TensorType>(dtype_param.out_type);
  auto out_shape = hsigmoid_grad_func_impl.InferShape(prim, {grad, x});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = hsigmoid_grad_func_impl.InferType(prim, {grad, x});
  ASSERT_TRUE(*out_dtype == *expect_type);
}

namespace {
auto HSigmoidGradOpTypeCases = testing::ValuesIn({
  EltwiseGradOpTypeParams{kFloat16, kFloat16, kFloat16},
  EltwiseGradOpTypeParams{kFloat32, kFloat32, kFloat32},
  EltwiseGradOpTypeParams{kBFloat16, kBFloat16, kBFloat16},
});
}

OP_FUNC_IMPL_SIMPLEINFER_TEST_DECLARE(HSigmoidGrad, MultiInputOpParams);
OP_FUNC_IMPL_SIMPLEINFER_TEST_CASES(
  HSigmoidGrad, testing::Values(MultiInputOpParams{{{2, 3}, {2, 3}}, {kFloat16, kFloat16}, {{2, 3}}, {kFloat16}, {}},
                                MultiInputOpParams{{{2, 3}, {2, 3}}, {kFloat32, kFloat32}, {{2, 3}}, {kFloat32}, {}},
                                MultiInputOpParams{{{2, 3}, {2, 3}}, {kBFloat16, kBFloat16}, {{2, 3}}, {kBFloat16}, {}}));

INSTANTIATE_TEST_CASE_P(TestHSigmoidGrad, TestHSigmoidGrad,
                        testing::Combine(EltwiseGradDynShapeTestCases, HSigmoidGradOpTypeCases));
}  // namespace ops
}  // namespace mindspore
