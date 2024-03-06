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
#include <vector>
#include <tuple>
#include "common/common_test.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/primal_attr.h"
#include "mindapi/base/shape_vector.h"
#include "test_value_utils.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/solve_triangular.h"
#include "include/c_api/ms/base/types.h"

namespace mindspore {
namespace ops {

struct SolveTriangularShape {
  ShapeVector a_shape;
  ShapeVector b_shape;
  ValuePtr trans;
  ValuePtr lower;
  ValuePtr unit_diagonal;
  ShapeVector out_shape;
};

struct SolveTriangularType {
  TypePtr a_type;
  TypePtr b_type;
  TypePtr out_type;
};

class TestSolveTriangular : public TestOps,
                            public testing::WithParamInterface<std::tuple<SolveTriangularShape, SolveTriangularType>> {
};

TEST_P(TestSolveTriangular, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  SolveTriangularFuncImpl solve_triangular_func_impl;
  auto primitive = std::make_shared<Primitive>("SolveTriangular");
  ASSERT_NE(primitive, nullptr);
  auto a = std::make_shared<abstract::AbstractTensor>(type_param.a_type, shape_param.a_shape);
  ASSERT_NE(a, nullptr);
  auto b = std::make_shared<abstract::AbstractTensor>(type_param.b_type, shape_param.b_shape);
  ASSERT_NE(b, nullptr);
  auto trans = shape_param.trans->ToAbstract();
  auto lower = shape_param.lower->ToAbstract();
  auto unit_diagonal = shape_param.unit_diagonal->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {a, b, trans, lower, unit_diagonal};

  // expect output
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_dtype = std::make_shared<TensorType>(type_param.out_type);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = solve_triangular_func_impl.InferShape(primitive, input_args);
  auto out_dtype = solve_triangular_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto solve_triangular_shape_cases = testing::Values(
  SolveTriangularShape{
    {5, 5}, {5}, CreateScalar<int64_t>(2), CreateScalar<bool>(false), CreateScalar<bool>(false), {5}},
  SolveTriangularShape{
    {5, 5}, {5, 5}, CreateScalar<int64_t>(1), CreateScalar<bool>(true), CreateScalar<bool>(false), {5, 5}},
  SolveTriangularShape{
    {3, 5, 5}, {3, 5}, CreateScalar<int64_t>(0), CreateScalar<bool>(false), CreateScalar<bool>(true), {3, 5}},
  SolveTriangularShape{
    {-1, -1}, {-1}, CreateScalar<int64_t>(2), CreateScalar<bool>(false), CreateScalar<bool>(false), {-1}},
  SolveTriangularShape{
    {-1, -1}, {5}, CreateScalar<int64_t>(1), CreateScalar<bool>(false), CreateScalar<bool>(false), {5}},
  SolveTriangularShape{
    {-2}, {-1, -1}, CreateScalar<int64_t>(0), CreateScalar<bool>(false), CreateScalar<bool>(false), {-1, -1}},
  SolveTriangularShape{
    {-2}, {-2}, CreateScalar<int64_t>(1), CreateScalar<bool>(false), CreateScalar<bool>(false), {-2}},
  SolveTriangularShape{
    {-2}, {5, 6}, CreateScalar<int64_t>(2), CreateScalar<bool>(false), CreateScalar<bool>(false), {5, 6}});

auto solve_triangular_type_cases = testing::ValuesIn(
  {SolveTriangularType{kFloat16, kFloat16, kFloat16}, SolveTriangularType{kFloat32, kFloat32, kFloat32},
   SolveTriangularType{kFloat64, kFloat64, kFloat64}, SolveTriangularType{kComplex64, kComplex64, kComplex64},
   SolveTriangularType{kComplex128, kComplex128, kComplex128}, SolveTriangularType{kInt8, kInt8, kFloat32},
   SolveTriangularType{kInt16, kInt16, kFloat32}, SolveTriangularType{kInt32, kInt32, kFloat32},
   SolveTriangularType{kInt64, kInt64, kFloat64}});

INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, TestSolveTriangular,
                        testing::Combine(solve_triangular_shape_cases, solve_triangular_type_cases));

}  // namespace ops
}  // namespace mindspore
