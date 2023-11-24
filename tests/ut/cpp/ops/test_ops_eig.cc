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
#include "ops/ops_func_impl/eig.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct EigShape {
  ShapeVector x_shape;
  ValuePtr compute_v;
  ShapeVector out_shape_u;
  ShapeVector out_shape_v;
};

struct EigDtype {
  TypePtr input_dtype;
  TypePtr output_dtype;
};

class TestEig : public TestOps, public testing::WithParamInterface<std::tuple<EigShape, EigDtype>> {};

TEST_P(TestEig, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  EigFuncImpl eig_func_impl;
  auto prim = std::make_shared<Primitive>("Eig");

  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.input_dtype, shape_param.x_shape);
  auto compute_v = shape_param.compute_v->ToAbstract();

  std::vector<BaseShapePtr> shapes_list = {std::make_shared<abstract::Shape>(shape_param.out_shape_u),
                                           std::make_shared<abstract::Shape>(shape_param.out_shape_v)};
  auto expect_shape = std::make_shared<abstract::TupleShape>(std::vector<BaseShapePtr>{shapes_list});
  auto dtype_out = std::make_shared<TensorType>(dtype_param.output_dtype);
  auto expect_dtype = std::make_shared<Tuple>(std::vector<TypePtr>{dtype_out, dtype_out});

  auto out_shape = eig_func_impl.InferShape(prim, {x, compute_v});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = eig_func_impl.InferType(prim, {x, compute_v});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto EigDynTestCase = testing::ValuesIn({
  /* static */
  EigShape{{20, 30, 30}, CreateScalar(true), {20, 30}, {20, 30, 30}},
  EigShape{{2, 4, 6, 8, 8}, CreateScalar(kValueAny), {2, 4, 6, 8}, {-2}},
  EigShape{{288, 288}, CreateScalar(false), {288}, {}},
  EigShape{{2, 4, 6, 8, 8}, CreateScalar(false), {2, 4, 6, 8}, {}},
  /* dynamic shape */
  EigShape{{-1, -1, -1}, CreateScalar(true), {-1, -1}, {-1, -1, -1}},
  EigShape{{-1, 2, -1, -1}, CreateScalar(true), {-1, 2, -1}, {-1, 2, -1, -1}},
  EigShape{{2, 2, 4, -1}, CreateScalar(true), {2, 2, 4}, {2, 2, 4, -1}},
  EigShape{{-1, 6, 6}, CreateScalar(true), {-1, 6}, {-1, 6, 6}},
  EigShape{{-1, 9}, CreateScalar(true), {9}, {-1, 9}},
  EigShape{{-2}, CreateScalar(true), {-2}, {-2}},
  EigShape{{-1, -1, -1, -1, -1}, CreateScalar(false), {-1, -1, -1, -1}, {}},
  EigShape{{-1, 2, -1, -1}, CreateScalar(false), {-1, 2, -1}, {}},
  EigShape{{2, 2, 4, -1}, CreateScalar(kValueAny), {2, 2, 4}, {-2}},
  EigShape{{-1, 6, 6}, CreateScalar(false), {-1, 6}, {}},
  EigShape{{99, -1}, CreateScalar(kValueAny), {99}, {-2}},
  /* dynamic rank */
  EigShape{{-2}, CreateScalar(false), {-2}, {}},
  EigShape{{-2}, CreateScalar(true), {-2}, {-2}},
  EigShape{{-2}, CreateScalar(kValueAny), {-2}, {-2}},
});

auto EigOpTypeCases = testing::ValuesIn({
  EigDtype{kFloat32, kComplex64},
  EigDtype{kFloat64, kComplex128},
  EigDtype{kComplex64, kComplex64},
  EigDtype{kComplex128, kComplex128},
});

INSTANTIATE_TEST_CASE_P(TestEig, TestEig, testing::Combine(EigDynTestCase, EigOpTypeCases));
}  // namespace ops
}  // namespace mindspore
