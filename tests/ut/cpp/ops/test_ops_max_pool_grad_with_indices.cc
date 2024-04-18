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

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/op_name.h"
#include "ops/ops_func_impl/max_pool_grad_with_indices.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
struct MaxPoolGradWithIndicesParams {
  ShapeVector x_shape;
  TypePtr x_dtype;
  ShapeVector out_shape;
  TypePtr out_dtype;
};

class TestMaxPoolGradWithIndices : public TestOps, public testing::WithParamInterface<MaxPoolGradWithIndicesParams> {};

TEST_P(TestMaxPoolGradWithIndices, dyn_shape) {
  const auto &param = GetParam();
  auto max_pool_grad_with_indices_func_impl = std::make_shared<MaxPoolGradWithIndicesFuncImpl>();
  auto prim = std::make_shared<Primitive>("MaxPoolGradWithIndices");

  auto x = std::make_shared<abstract::AbstractTensor>(param.x_dtype, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_dtype);
  auto infer_shape = max_pool_grad_with_indices_func_impl->InferShape(prim, {x});
  ASSERT_NE(infer_shape, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  auto infer_type = max_pool_grad_with_indices_func_impl->InferType(prim, {x});
  ASSERT_NE(infer_type, nullptr);
  ASSERT_TRUE(*infer_type == *expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestMaxPoolGradWithIndicesGroup, TestMaxPoolGradWithIndices,
  testing::Values(MaxPoolGradWithIndicesParams{{1, 3, 5, 5}, kFloat16, {1, 3, 5, 5}, kFloat16},
                  MaxPoolGradWithIndicesParams{{1, 3, -1, -1}, kFloat32, {1, 3, -1, -1}, kFloat32},
                  MaxPoolGradWithIndicesParams{{-1, -1, -1, -1}, kFloat16, {-1, -1, -1, -1}, kFloat16},
                  MaxPoolGradWithIndicesParams{{-2}, kFloat32, {-1, -1, -1, -1}, kFloat32}));

}  // namespace ops
}  // namespace mindspore
