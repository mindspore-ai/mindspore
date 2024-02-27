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
#include "ops/ops_func_impl/clamp_scalar.h"
#include "ops/ops_func_impl/clamp_tensor.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct ClampShapeParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector min_shape;
  TypePtr min_type;
  ShapeVector max_shape;
  TypePtr max_type;
};

class TestClampTensor : public TestOps, public testing::WithParamInterface<ClampShapeParams> {};

TEST_P(TestClampTensor, clamp_dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto min = std::make_shared<abstract::AbstractTensor>(param.min_type, param.min_shape);
  auto max = std::make_shared<abstract::AbstractTensor>(param.max_type, param.max_shape);


  auto expect_shape = std::make_shared<abstract::Shape>(param.x_shape);
  auto expect_type = std::make_shared<TensorType>(param.x_type);

  ClampTensorFuncImpl clamp_func_impl;
  auto prim = std::make_shared<Primitive>("ClampTensor");

  auto out_dtype = clamp_func_impl.InferType(prim, {x, min, max});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = clamp_func_impl.InferShape(prim, {x, min, max});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestClampTensor, TestClampTensor,
  testing::Values(ClampShapeParams{{3, 4, 5}, kFloat32, {3, 4, 1}, kFloat32, {3, 4, 1}, kFloat32},
                  ClampShapeParams{{3, 4, 5}, kInt64, {}, kFloat32, {}, kFloat32},
                  ClampShapeParams{{3, 4, 5}, kInt64, {}, kInt64, {}, kInt64},
                  ClampShapeParams{{3, 4, 5}, kFloat16, {}, kFloat32, {}, kFloat32},
                  ClampShapeParams{{-1, -1, -1}, kInt32, {}, kFloat32, {}, kFloat32},
                  ClampShapeParams{{-2}, kFloat32, {}, kFloat32, {}, kFloat32}));


class TestClampScalar : public TestOps, public testing::WithParamInterface<ClampShapeParams> {};

TEST_P(TestClampScalar, clamp_dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto min = std::make_shared<abstract::AbstractScalar>(kValueAny, param.min_type);
  auto max = std::make_shared<abstract::AbstractScalar>(kValueAny, param.max_type);


  auto expect_shape = std::make_shared<abstract::Shape>(param.x_shape);
  auto expect_type = std::make_shared<TensorType>(param.x_type);

  ClampScalarFuncImpl clamp_func_impl;
  auto prim = std::make_shared<Primitive>("ClampScalar");

  auto out_dtype = clamp_func_impl.InferType(prim, {x, min, max});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = clamp_func_impl.InferShape(prim, {x, min, max});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestClampScalar, TestClampScalar,
  testing::Values(ClampShapeParams{{3, 4, 5}, kFloat32, {}, kFloat32, {}, kFloat32},
                  ClampShapeParams{{3, 4, 5}, kInt64, {}, kFloat32, {}, kFloat32},
                  ClampShapeParams{{3, 4, 5}, kInt64, {}, kInt64, {}, kInt64},
                  ClampShapeParams{{3, 4, 5}, kFloat16, {}, kFloat32, {}, kFloat32},
                  ClampShapeParams{{-1, -1, -1}, kInt32, {}, kFloat32, {}, kFloat32},
                  ClampShapeParams{{-2}, kFloat32, {}, kFloat32, {}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
