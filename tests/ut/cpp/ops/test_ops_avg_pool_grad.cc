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

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/op_name.h"
#include "ops/ops_func_impl/avg_pool_grad.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
struct AvgPoolGradParams {
  ShapeVector x_shape;
  TypePtr x_dtype;
  ShapeVector out_shape;
};

class TestAvgPoolGrad : public TestOps, public testing::WithParamInterface<AvgPoolGradParams> {};

TEST_P(TestAvgPoolGrad, dyn_shape) {
  const auto &param = GetParam();
  auto avg_pool_grad_func_impl = std::make_shared<AvgPoolGradFuncImpl>();
  auto prim = std::make_shared<Primitive>("AvgPoolGrad");

  auto x = std::make_shared<abstract::AbstractTensor>(param.x_dtype, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto infer_shape = avg_pool_grad_func_impl->InferShape(prim, {x});
  ASSERT_NE(infer_shape, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(TestAvgPoolGradGroup, TestAvgPoolGrad,
                        testing::Values(AvgPoolGradParams{{1, 3, 5, 5}, kFloat16, {1, 3, 5, 5}},
                                        AvgPoolGradParams{{1, 3, -1, -1}, kFloat32, {1, 3, -1, -1}},
                                        AvgPoolGradParams{{-1, -1, -1, -1}, kFloat64, {-1, -1, -1, -1}},
                                        AvgPoolGradParams{{-2}, kFloat64, {-2}}));

}  // namespace ops
}  // namespace mindspore
