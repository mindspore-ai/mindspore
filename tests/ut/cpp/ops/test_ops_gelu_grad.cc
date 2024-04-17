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
#include "ops/test_ops.h"
#include "ops/ops_func_impl/gelu_grad.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct GeLUGradShape {
  ShapeVector dy_shape;
  ShapeVector x_shape;
  ShapeVector y_shape;
  ShapeVector out_shape;
};
struct GeLUGradType {
  TypePtr dy_type;
  TypePtr x_type;
  TypePtr y_type;
  TypePtr out_type;
};

class TestGeLUGrad : public TestOps, public testing::WithParamInterface<std::tuple<GeLUGradShape, GeLUGradType>> {};

TEST_P(TestGeLUGrad, gelu_grad_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  GeLUGradFuncImpl gelu_grad_func_impl;
  auto prim = std::make_shared<Primitive>("GeLUGrad");
  auto dy = std::make_shared<abstract::AbstractTensor>(dtype_param.dy_type, shape_param.dy_shape);
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  auto y = std::make_shared<abstract::AbstractTensor>(dtype_param.y_type, shape_param.y_shape);
  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);

  auto out_shape = gelu_grad_func_impl.InferShape(prim, {dy, x, y});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = gelu_grad_func_impl.InferType(prim, {dy, x, y});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto GeLUGradOpShapeTestCases = testing::ValuesIn({
  /* static */
  GeLUGradShape{{2, 3, 4}, {2, 3, 4}, {2, 3, 4}, {2, 3, 4}},
  /* dynamic shape */
  GeLUGradShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}},
  GeLUGradShape{{2, 1, 1, -1}, {2, 1, 1, -1}, {2, 1, 1, -1}, {2, 1, 1, -1}},
  GeLUGradShape{{5, 3, -1, 2, -1}, {5, 3, -1, 2, -1}, {5, 3, -1, 2, -1}, {5, 3, -1, 2, -1}},
  GeLUGradShape{
    {5, 3, -1, 2, 1, 4, 7, 4}, {5, 3, -1, 2, 1, 4, 7, 4}, {5, 3, -1, 2, 1, 4, 7, 4}, {5, 3, -1, 2, 1, 4, 7, 4}},
  /* dynamic rank */
  GeLUGradShape{{-2}, {-2}, {-2}, {-2}},
});

auto GeLUGradOpTypeTestCases = testing::ValuesIn({
  GeLUGradType{kFloat16, kFloat16, kFloat16, kFloat16},
  GeLUGradType{kFloat32, kFloat32, kFloat32, kFloat32},
  GeLUGradType{kFloat64, kFloat64, kFloat64, kFloat64},
});

INSTANTIATE_TEST_CASE_P(TestGeLUGrad, TestGeLUGrad,
                        testing::Combine(GeLUGradOpShapeTestCases, GeLUGradOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
