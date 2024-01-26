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
#include "ops/ops_func_impl/correlate.h"
#include "include/c_api/ms/base/types.h"

namespace mindspore {
namespace ops {

struct CorrelateShape {
  ShapeVector a_shape;
  ShapeVector v_shape;
  ValuePtr mode;
  ShapeVector out_shape;
};

struct CorrelateType {
  TypePtr a_type;
  TypePtr v_type;
  TypePtr out_type;
};

class TestCorrelate : public TestOps, public testing::WithParamInterface<std::tuple<CorrelateShape, CorrelateType>> {};

TEST_P(TestCorrelate, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  CorrelateFuncImpl correlate_func_impl;
  auto primitive = std::make_shared<Primitive>("Correlate");
  ASSERT_NE(primitive, nullptr);
  auto a = std::make_shared<abstract::AbstractTensor>(type_param.a_type, shape_param.a_shape);
  ASSERT_NE(a, nullptr);
  auto v = std::make_shared<abstract::AbstractTensor>(type_param.v_type, shape_param.v_shape);
  ASSERT_NE(v, nullptr);
  auto mode = shape_param.mode->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {a, v, mode};

  // expect output
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_dtype = std::make_shared<TensorType>(type_param.out_type);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = correlate_func_impl.InferShape(primitive, input_args);
  auto out_dtype = correlate_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto correlate_shape_cases = testing::Values(
  CorrelateShape{{5}, {7}, CreateScalar<int64_t>(2), {3}}, CorrelateShape{{7}, {5}, CreateScalar<int64_t>(1), {7}},
  CorrelateShape{{5}, {7}, CreateScalar<int64_t>(3), {11}}, CorrelateShape{{-1}, {-1}, CreateScalar<int64_t>(2), {-1}},
  CorrelateShape{{-1}, {-1}, CreateScalar<int64_t>(1), {-1}},
  CorrelateShape{{-1}, {-1}, CreateScalar<int64_t>(3), {-1}},
  CorrelateShape{{-2}, {-2}, CreateScalar<int64_t>(2), {-1}},
  CorrelateShape{{-2}, {-2}, CreateScalar<int64_t>(1), {-1}},
  CorrelateShape{{-2}, {-2}, CreateScalar<int64_t>(3), {-1}});

auto correlate_type_cases =
  testing::ValuesIn({CorrelateType{kFloat16, kFloat16, kFloat16}, CorrelateType{kFloat32, kFloat32, kFloat32},
                     CorrelateType{kFloat64, kFloat64, kFloat64}, CorrelateType{kComplex64, kComplex64, kComplex64},
                     CorrelateType{kComplex128, kComplex128, kComplex128}, CorrelateType{kInt8, kInt8, kFloat32},
                     CorrelateType{kInt16, kInt16, kFloat32}, CorrelateType{kInt32, kInt32, kFloat32},
                     CorrelateType{kInt64, kInt64, kFloat64}});

INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, TestCorrelate, testing::Combine(correlate_shape_cases, correlate_type_cases));

}  // namespace ops
}  // namespace mindspore
