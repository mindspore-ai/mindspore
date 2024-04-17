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
#include "ops/ops_func_impl/qr.h"

namespace mindspore {
namespace ops {
struct QrShape {
  ShapeVector x_shape;
  ValuePtr full_matrices;
  ShapeVector q_shape;
  ShapeVector r_shape;
};

struct QrType {
  TypePtr x_type;
  TypePtr q_type;
  TypePtr r_type;
};

class TestQR : public TestOps, public testing::WithParamInterface<std::tuple<QrShape, QrType>> {};

TEST_P(TestQR, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  QrFuncImpl qr_func_impl;
  auto primitive = std::make_shared<Primitive>("Qr");
  ASSERT_NE(primitive, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(type_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto full_matrices = shape_param.full_matrices->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {x, full_matrices};

  // expect output
  std::vector<BaseShapePtr> shapes_list = {std::make_shared<abstract::Shape>(shape_param.q_shape),
                                           std::make_shared<abstract::Shape>(shape_param.r_shape)};
  auto expect_shape = std::make_shared<abstract::TupleShape>(shapes_list);
  ASSERT_NE(expect_shape, nullptr);
  std::vector<TypePtr> types_list = {std::make_shared<TensorType>(type_param.q_type),
                                     std::make_shared<TensorType>(type_param.r_type)};
  auto expect_dtype = std::make_shared<Tuple>(types_list);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = qr_func_impl.InferShape(primitive, input_args);
  auto out_dtype = qr_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto qr_shape_cases = testing::Values(
  QrShape{{5, 4}, CreateScalar(true), {5, 5}, {5, 4}},
  QrShape{{5, 4}, CreateScalar(false), {5, 4}, {4, 4}},
  QrShape{{-1, -1, -1}, CreateScalar(true), {-1, -1, -1}, {-1, -1, -1}},
  QrShape{{-1, -1, -1}, CreateScalar(false), {-1, -1, -1}, {-1, -1, -1}},
  QrShape{{-2}, CreateScalar(true), {-2}, {-2}},
  QrShape{{-2}, CreateScalar(false), {-2}, {-2},
});

auto qr_type_cases = testing::ValuesIn({
  QrType{kFloat16, kFloat16, kFloat16},
  QrType{kFloat32, kFloat32, kFloat32},
  QrType{kFloat64, kFloat64, kFloat64},
  QrType{kComplex64, kComplex64, kComplex64},
  QrType{kComplex128, kComplex128, kComplex128},
});

INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, TestQR, testing::Combine(qr_shape_cases, qr_type_cases));

}  // namespace ops
}  // namespace mindspore
