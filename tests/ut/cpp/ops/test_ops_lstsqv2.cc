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
#include "ops/ops_func_impl/lstsq_v2.h"
#include "include/c_api/ms/base/types.h"

namespace mindspore {
namespace ops {

struct LstsqV2Shape {
  ShapeVector a_shape;
  ShapeVector b_shape;
  ValuePtr driver;
  ShapeVector solution_shape;
  ShapeVector residual_shape;
  ShapeVector rank_shape;
  ShapeVector singular_value_shape;
};

struct LstsqV2Type {
  TypePtr a_type;
  TypePtr b_type;
  TypePtr solution_type;
  TypePtr residual_type;
  TypePtr rank_type;
  TypePtr singular_value_type;
};

class TestLstsqV2 : public TestOps, public testing::WithParamInterface<std::tuple<LstsqV2Shape, LstsqV2Type>> {};

TEST_P(TestLstsqV2, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  LstsqV2FuncImpl lstsqv2_func_impl;
  auto primitive = std::make_shared<Primitive>("LstsqV2");
  ASSERT_NE(primitive, nullptr);
  auto a = std::make_shared<abstract::AbstractTensor>(type_param.a_type, shape_param.a_shape);
  ASSERT_NE(a, nullptr);
  auto b = std::make_shared<abstract::AbstractTensor>(type_param.b_type, shape_param.b_shape);
  ASSERT_NE(b, nullptr);
  auto driver = shape_param.driver->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {a, b, driver};

  // expect output
  std::vector<BaseShapePtr> shapes_list = {
    std::make_shared<abstract::Shape>(shape_param.solution_shape),
    std::make_shared<abstract::Shape>(shape_param.residual_shape),
    std::make_shared<abstract::Shape>(shape_param.rank_shape),
    std::make_shared<abstract::Shape>(shape_param.singular_value_shape),
  };
  auto expect_shape = std::make_shared<abstract::TupleShape>(shapes_list);
  ASSERT_NE(expect_shape, nullptr);
  std::vector<TypePtr> types_list = {
    std::make_shared<TensorType>(type_param.solution_type), std::make_shared<TensorType>(type_param.residual_type),
    std::make_shared<TensorType>(type_param.rank_type), std::make_shared<TensorType>(type_param.singular_value_type)};
  auto expect_dtype = std::make_shared<Tuple>(types_list);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = lstsqv2_func_impl.InferShape(primitive, input_args);
  auto out_dtype = lstsqv2_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto lstsqv2_shape_cases = testing::Values(
  LstsqV2Shape{{5, 6, 7}, {1, 6, 2}, CreateScalar<int64_t>(0), {5, 7, 2}, {0}, {0}, {0}},
  LstsqV2Shape{{5, 7, 6}, {1, 7, 2}, CreateScalar<int64_t>(0), {5, 6, 2}, {5, 2}, {0}, {0}},
  LstsqV2Shape{{1, 7, 6}, {5, 7, 2}, CreateScalar<int64_t>(1), {5, 6, 2}, {0}, {1}, {0}},
  LstsqV2Shape{{5, 6, 7}, {1, 6, 2}, CreateScalar<int64_t>(2), {5, 7, 2}, {0}, {5}, {5, 6}},
  LstsqV2Shape{{5, 7, 6}, {1, 7, 2}, CreateScalar<int64_t>(3), {5, 6, 2}, {5, 2}, {5}, {5, 6}},
  LstsqV2Shape{{-1, -1, -1}, {-1, -1, -1}, CreateScalar<int64_t>(3), {-1, -1, -1}, {-1, -1}, {-1}, {-1, -1}},
  LstsqV2Shape{{-2}, {-2}, CreateScalar<int64_t>(3), {-2}, {-2}, {-2}, {-2}});

auto lstsqv2_type_cases =
  testing::ValuesIn({LstsqV2Type{kFloat32, kFloat32, kFloat32, kFloat32, kInt64, kFloat32},
                     LstsqV2Type{kFloat64, kFloat64, kFloat64, kFloat64, kInt64, kFloat64},
                     LstsqV2Type{kComplex64, kComplex64, kComplex64, kFloat32, kInt64, kFloat32},
                     LstsqV2Type{kComplex128, kComplex128, kComplex128, kFloat64, kInt64, kFloat64}});

INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, TestLstsqV2, testing::Combine(lstsqv2_shape_cases, lstsqv2_type_cases));

}  // namespace ops
}  // namespace mindspore
