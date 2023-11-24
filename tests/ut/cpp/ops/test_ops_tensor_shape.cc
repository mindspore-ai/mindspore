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
#include "ops/ops_func_impl/tensor_shape.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {
struct TensorShapeParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector out_shape;
};

class TestTensorShape : public TestOps, public testing::WithParamInterface<TensorShapeParams> {};

TEST_P(TestTensorShape, dyn_shape) {
  const auto &param = GetParam();

  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(kInt64);
  DoFuncImplInferAndCompare<TensorShapeFuncImpl>("TensorShape", abstract::AbstractBasePtrList{x}, expect_shape,
                                                 expect_type);
}

INSTANTIATE_TEST_CASE_P(TestTensorShape, TestTensorShape,
                        testing::Values(TensorShapeParams{{2, 3, 4}, kFloat32, {3}},
                                        TensorShapeParams{{3, -1, 4}, kFloat32, {3}},
                                        TensorShapeParams{{-2}, kFloat32, {-1}}));
}  // namespace mindspore::ops
