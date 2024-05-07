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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/bitwise_and_scalar.h"
#include "ops/ops_func_impl/bitwise_and_tensor.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct BitwiseAndShapeParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector y_shape;
  TypePtr y_type;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestBitwiseAndTensor : public TestOps, public testing::WithParamInterface<BitwiseAndShapeParams> {};

TEST_P(TestBitwiseAndTensor, bitwise_and_dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto y = std::make_shared<abstract::AbstractTensor>(param.y_type, param.y_shape);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.x_shape);
  auto expect_type = std::make_shared<TensorType>(param.x_type);

  BitwiseAndTensorFuncImpl bitwise_and_func_impl;
  auto prim = std::make_shared<Primitive>("BitwiseAndTensor");

  auto out_dtype = bitwise_and_func_impl.InferType(prim, {x, y});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = bitwise_and_func_impl.InferShape(prim, {x, y});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestBitwiseAndTensor, TestBitwiseAndTensor,
  testing::Values(BitwiseAndShapeParams{{3, 4, 5}, kInt32, {3, 4, 5}, kInt32, {3, 4, 5}, kInt32},
                  BitwiseAndShapeParams{{3, 5}, kInt64, {3, 5}, kInt64, {3, 5}, kInt64},
                  BitwiseAndShapeParams{{3, 4, 5}, kInt16, {1}, kInt16, {3, 4, 5}, kInt16},
                  BitwiseAndShapeParams{{3, 4}, kInt8, {3, 4}, kInt8, {3, 4}, kInt8},
                  BitwiseAndShapeParams{{-1, -1}, kInt32, {-1, -1}, kInt32, {-1, -1}, kInt32},
                  BitwiseAndShapeParams{{-2}, kInt8, {-2}, kInt8, {-2}, kInt8}));


class TestBitwiseAndScalar : public TestOps, public testing::WithParamInterface<BitwiseAndShapeParams> {};

TEST_P(TestBitwiseAndScalar, bitwise_and_dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto y = std::make_shared<abstract::AbstractScalar>(kValueAny, param.y_type);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.x_shape);
  auto expect_type = std::make_shared<TensorType>(param.x_type);

  BitwiseAndScalarFuncImpl bitwise_and_func_impl;
  auto prim = std::make_shared<Primitive>("BitwiseAndScalar");

  auto out_dtype = bitwise_and_func_impl.InferType(prim, {x, y});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = bitwise_and_func_impl.InferShape(prim, {x, y});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestBitwiseAndScalar, TestBitwiseAndScalar,
  testing::Values(BitwiseAndShapeParams{{3, 4, 5}, kInt32, {}, kInt32, {3, 4, 5}, kInt32},
                  BitwiseAndShapeParams{{3, 5}, kInt16, {}, kInt16, {3, 5}, kInt16},
                  BitwiseAndShapeParams{{3, 4, 5}, kInt64, {}, kInt64, {3, 4, 5}, kInt64},
                  BitwiseAndShapeParams{{-1, -1, -1}, kInt8, {}, kInt8, {-1, -1, -1}, kInt8},
                  BitwiseAndShapeParams{{-2}, kInt32, {}, kInt32, {-2}, kInt32}));
}  // namespace ops
}  // namespace mindspore
