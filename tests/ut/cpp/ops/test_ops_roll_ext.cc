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
#include "ops/ops_func_impl/roll.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_name.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct RollExtParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr shifts;
  ValuePtr dims;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestRollExt : public TestOps, public testing::WithParamInterface<RollExtParams> {};

TEST_P(TestRollExt, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto shifts = param.shifts->ToAbstract();
  auto dims = param.dims->ToAbstract();
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);

  RollFuncImpl roll_func_impl;
  auto prim = std::make_shared<Primitive>("RollExt");

  auto out_dtype = roll_func_impl.InferType(prim, {x, shifts, dims});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = roll_func_impl.InferShape(prim, {x, shifts, dims});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

class TestRollExtSimpleInfer : public TestOps, public testing::WithParamInterface<RollExtParams> {};

TEST_P(TestRollExtSimpleInfer, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<tensor::BaseTensor>(param.x_type->type_id(), param.x_shape);
  ValuePtrList input_values;
  input_values.push_back(std::move(x));
  input_values.push_back(std::move(param.shifts));
  input_values.push_back(std::move(param.dims));

  RollFuncImpl roll_func_impl;
  auto prim = std::make_shared<Primitive>("RollExt");

  auto expect_shape = ShapeArray{param.out_shape};
  auto expect_type = TypePtrList{param.out_type};

  auto output_shape = roll_func_impl.InferShape(prim, input_values);
  auto output_type = roll_func_impl.InferType(prim, input_values);

  ShapeCompare(output_shape, expect_shape);
  TypeCompare(output_type, expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestRollExtGroup, TestRollExt,
  testing::Values(RollExtParams{{2, 3}, kFloat32, CreateTuple({1}), CreateTuple({1}), {2, 3}, kFloat32},
                  RollExtParams{{-1, 2, 3}, kFloat16, CreateTuple({0}), CreateTuple({0}), {-1, 2, 3}, kFloat16},
                  RollExtParams{{-1, -1}, kInt8, CreateTuple({1}), CreateTuple({1}), {-1, -1}, kInt8},
                  RollExtParams{{-2}, kUInt64, CreateTuple({1}), CreateTuple({1}), {-2}, kUInt64}));

INSTANTIATE_TEST_CASE_P(
  TestRollExtGroup, TestRollExtSimpleInfer,
  testing::Values(RollExtParams{{2, 3}, kFloat32, CreateTuple({1}), CreateTuple({1}), {2, 3}, kFloat32},
                  RollExtParams{{4, 2, 3}, kFloat16, CreateTuple({0}), CreateTuple({0}), {4, 2, 3}, kFloat16},
                  RollExtParams{{3, 4, 5, 6}, kInt8, CreateTuple({1}), CreateTuple({1}), {3, 4, 5, 6}, kInt8},
                  RollExtParams{
                    {3, 4, 5, 6, 7}, kUInt64, CreateTuple({1}), CreateTuple({1}), {3, 4, 5, 6, 7}, kUInt64}));
}  // namespace ops
}  // namespace mindspore
