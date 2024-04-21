/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <memory>
#include "ops/ops_func_impl/stack_ext.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {
struct StackParams {
  bool dynamic_len;
  ShapeArray x_shapes;
  TypePtr x_type;
  ValuePtr axis_value;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestStack : public TestOps, public testing::WithParamInterface<StackParams> {};

TEST_P(TestStack, dyn_shape) {
  const auto &param = GetParam();

  AbstractBasePtrList inputs;
  inputs.reserve(param.x_shapes.size());
  for (auto x_shape : param.x_shapes) {
    auto input = std::make_shared<abstract::AbstractTensor>(param.x_type, x_shape);
    ASSERT_NE(input, nullptr);
    inputs.push_back(input);
  }

  auto tuple_x = std::make_shared<abstract::AbstractTuple>(inputs);
  ASSERT_NE(tuple_x, nullptr);
  if (param.dynamic_len) {
    tuple_x->CheckAndConvertToDynamicLenSequence();
  }

  auto axis = std::make_shared<abstract::AbstractScalar>(param.axis_value, kInt64);
  ASSERT_NE(axis, nullptr);

  auto expect_shape = std::make_shared<abstract::TensorShape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<StackExtFuncImpl>("StackExt", abstract::AbstractBasePtrList{tuple_x, axis}, expect_shape,
                                           expect_type);
}

INSTANTIATE_TEST_CASE_P(TestStackGroup, TestStack,
                        testing::Values(
                          // Normal.
                          StackParams{false, {{3, 4}, {3, 4}}, kFloat32, CreatePyInt(1), {3, 2, 4}, kFloat32},
                          StackParams{false, {{3, 4}, {3, 4}}, kFloat32, kValueAny, {-1, -1, -1}, kFloat32},
                          StackParams{false, {{3, 4}, {3, 4}}, kFloat32, CreatePyInt(1), {3, 2, 4}, kFloat32},
                          StackParams{false, {{-1, 4}, {-1, 4}}, kFloat32, CreatePyInt(1), {-1, 2, 4}, kFloat32},
                          StackParams{false, {{-2}, {-2}}, kFloat32, CreatePyInt(1), {-2}, kFloat32},
                          StackParams{true, {{-2}}, kFloat32, CreatePyInt(1), {-2}, kFloat32},
                          StackParams{true, {{-2}}, kFloat32, kValueAny, {-2}, kFloat32},
                          // Static -> Dynamic.
                          StackParams{false, {{3, 4}, {3, 4}}, kFloat32, kValueAny, {-1, -1, -1}, kFloat32},
                          // The size of dynamic shape dim increase.
                          StackParams{false, {{-1, 4}, {3, -1}}, kFloat32, kValueAny, {-1, -1, -1}, kFloat32},
                          // Dynamic level(from light to heavy): static, dynamic shape, dynamic rank.
                          // Dynamic shrink from heavy case to lighter one.
                          StackParams{false, {{3, -1}, {-1, 4}}, kFloat32, CreatePyInt(1), {3, 2, 4}, kFloat32},
                          StackParams{false, {{-2}, {3, 4}}, kFloat32, CreatePyInt(1), {3, 2, 4}, kFloat32},
                          StackParams{false, {{-2}, {-1, 4}}, kFloat32, CreatePyInt(1), {-1, 2, 4}, kFloat32},
                          StackParams{true, {{3, 4}}, kFloat32, CreatePyInt(1), {3, -1, 4}, kFloat32},
                          StackParams{true, {{3, 4}}, kFloat32, CreatePyInt(0), {-1, 3, 4}, kFloat32},
                          StackParams{true, {{-1, 4}}, kFloat32, CreatePyInt(1), {-1, -1, 4}, kFloat32},
                          StackParams{true, {{3, 4}}, kFloat32, kValueAny, {-1, -1, -1}, kFloat32},
                          StackParams{true, {{-1, 4}}, kFloat32, kValueAny, {-1, -1, -1}, kFloat32}));
}  // namespace mindspore::ops
