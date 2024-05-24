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
#include "ops/ops_func_impl/sign.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct SignShapeParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestSign : public TestOps, public testing::WithParamInterface<SignShapeParams> {};

TEST_P(TestSign, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.out_type);

  SignFuncImpl sign_func_impl;
  auto prim = std::make_shared<Primitive>("Sign");

  auto out_dtype = sign_func_impl.InferType(prim, {x});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape = sign_func_impl.InferShape(prim, {x});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(TestSign, TestSign,
                        testing::Values(SignShapeParams{{5, 2},
                                                        kInt32,
                                                        {5, 2},
                                                        kInt32},
                                        SignShapeParams{{252, 5, 2},
                                                        kFloat32,
                                                        {252, 5, 2},
                                                        kFloat32}));

}  // namespace ops
}  // namespace mindspore