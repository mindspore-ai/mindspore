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
#include "ops/ops_func_impl/lstsq_v2_grad.h"
#include "include/c_api/ms/base/types.h"

namespace mindspore {
namespace ops {

struct LstsqV2GradShape {
  ShapeVector gx_shape;
  ShapeVector a_shape;
  ShapeVector b_shape;
  ShapeVector ga_shape;
  ShapeVector gb_shape;
};

struct LstsqV2GradType {
  TypePtr gx_type;
  TypePtr a_type;
  TypePtr b_type;
  TypePtr ga_type;
  TypePtr gb_type;
};

class TestLstsqV2Grad : public TestOps, public testing::WithParamInterface<std::tuple<LstsqV2GradShape, LstsqV2GradType>> {};

TEST_P(TestLstsqV2Grad, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  LstsqV2GradFuncImpl lstsqv2_grad_func_impl;
  auto primitive = std::make_shared<Primitive>("LstsqV2Grad");
  ASSERT_NE(primitive, nullptr);
  auto gx = std::make_shared<abstract::AbstractTensor>(type_param.gx_type, shape_param.gx_shape);
  ASSERT_NE(gx, nullptr);
  auto a = std::make_shared<abstract::AbstractTensor>(type_param.a_type, shape_param.a_shape);
  ASSERT_NE(a, nullptr);
  auto b = std::make_shared<abstract::AbstractTensor>(type_param.b_type, shape_param.b_shape);
  ASSERT_NE(b, nullptr);
  std::vector<AbstractBasePtr> input_args = {gx, a, b};

  // expect output
  std::vector<BaseShapePtr> shapes_list = {
    std::make_shared<abstract::Shape>(shape_param.ga_shape),
    std::make_shared<abstract::Shape>(shape_param.gb_shape),
  };
  auto expect_shape = std::make_shared<abstract::TupleShape>(shapes_list);
  ASSERT_NE(expect_shape, nullptr);
  std::vector<TypePtr> types_list = {
    std::make_shared<TensorType>(type_param.ga_type), std::make_shared<TensorType>(type_param.gb_type)};
  auto expect_dtype = std::make_shared<Tuple>(types_list);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = lstsqv2_grad_func_impl.InferShape(primitive, input_args);
  auto out_dtype = lstsqv2_grad_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto lstsqv2_grad_shape_cases = testing::Values(
  LstsqV2GradShape{{5, 6, 7}, {1, 6, 6}, {5, 6, 7}, {1, 6, 6}, {5, 6, 7}},
  LstsqV2GradShape{{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}},
  LstsqV2GradShape{{-2}, {-2}, {-2}, {-2}, {-2}});

auto lstsqv2_grad_type_cases =
  testing::ValuesIn({LstsqV2GradType{kFloat32, kFloat32, kFloat32, kFloat32, kFloat32},
                     LstsqV2GradType{kFloat64, kFloat64, kFloat64, kFloat64, kFloat64},
                     LstsqV2GradType{kComplex64, kComplex64, kComplex64, kComplex64, kComplex64},
                     LstsqV2GradType{kComplex128, kComplex128, kComplex128, kComplex128, kComplex128}});

INSTANTIATE_TEST_CASE_P(TestOpsFuncImpl, TestLstsqV2Grad, testing::Combine(lstsqv2_grad_shape_cases, lstsqv2_grad_type_cases));

}  // namespace ops
}  // namespace mindspore
