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
#include "ops/ops_func_impl/hsigmoid.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
class TestHSigmoid : public TestOps,
                     public testing::WithParamInterface<std::tuple<EltwiseOpShapeParams, EltwiseOpTypeParams>> {};

TEST_P(TestHSigmoid, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  HSigmoidFuncImpl hsigmoid_func_impl;
  auto prim = std::make_shared<Primitive>("HSigmoid");

  auto input_x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  ASSERT_NE(input_x, nullptr);
  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_type = std::make_shared<TensorType>(dtype_param.out_type);
  auto out_shape = hsigmoid_func_impl.InferShape(prim, {input_x});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = hsigmoid_func_impl.InferType(prim, {input_x});
  ASSERT_TRUE(*out_dtype == *expect_type);
}

namespace {
auto HSigmoidOpTypeCases = testing::ValuesIn({
  EltwiseOpTypeParams{kInt8, kInt8},
  EltwiseOpTypeParams{kInt16, kInt16},
  EltwiseOpTypeParams{kInt32, kInt32},
  EltwiseOpTypeParams{kInt64, kInt64},
  EltwiseOpTypeParams{kFloat16, kFloat16},
  EltwiseOpTypeParams{kFloat32, kFloat32},
  EltwiseOpTypeParams{kFloat64, kFloat64},
});
}

INSTANTIATE_TEST_CASE_P(TestHSigmoid, TestHSigmoid, testing::Combine(EltwiseDynShapeTestCases, HSigmoidOpTypeCases));
}  // namespace ops
}  // namespace mindspore
