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
#include "ops/ops_func_impl/gelu.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct GeLUShape {
  ShapeVector x_shape;
  ShapeVector out_shape;
};
struct GeLUType {
  TypePtr x_type;
  TypePtr out_type;
};

class TestGeLU : public TestOps, public testing::WithParamInterface<std::tuple<GeLUShape, GeLUType>> {};

TEST_P(TestGeLU, gelu_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  GeLUFuncImpl gelu_func_impl;
  auto prim = std::make_shared<Primitive>("GeLU");
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);

  auto out_shape = gelu_func_impl.InferShape(prim, {x});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = gelu_func_impl.InferType(prim, {x});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto GeLUOpShapeTestCases = testing::ValuesIn({
  /* static */
  GeLUShape{{2, 3, 4}, {2, 3, 4}},
  /* dynamic shape */
  GeLUShape{{-1}, {-1}},
  GeLUShape{{-1, 2, 4}, {-1, 2, 4}},
  GeLUShape{{5, 3, -1, 2, 1}, {5, 3, -1, 2, 1}},
  GeLUShape{{5, 3, -1, 2, 1, 4, 7, 4}, {5, 3, -1, 2, 1, 4, 7, 4}},
  /* dynamic rank */
  GeLUShape{{-2}, {-2}},
});

auto GeLUOpTypeTestCases = testing::ValuesIn({
  GeLUType{kFloat16, kFloat16},
  GeLUType{kFloat32, kFloat32},
  GeLUType{kFloat64, kFloat64},
});

INSTANTIATE_TEST_CASE_P(TestGeLU, TestGeLU, testing::Combine(GeLUOpShapeTestCases, GeLUOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
