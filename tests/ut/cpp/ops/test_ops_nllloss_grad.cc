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
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "common/common_test.h"
#include "include/backend/optimizer/helper.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/ops_func_impl/nllloss_grad.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "utils/ms_context.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
struct NLLLossGradOpParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector dy_shape;
  TypePtr dy_type;
  ShapeVector label_shape;
  TypePtr label_type;
  ShapeVector weight_shape;
  TypePtr weight_type;
  ShapeVector total_weight_shape;
  TypePtr total_weight_type;
  bool is_success;
  ShapeVector out_shape;
  TypePtr out_type;
};

const string kNameNLLLossGrad_ = "NLLLossGrad";

class TestNLLLossGrad : public TestOps, public testing::WithParamInterface<NLLLossGradOpParams> {};

TEST_P(TestNLLLossGrad, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto dy = std::make_shared<abstract::AbstractTensor>(param.dy_type, param.dy_shape);
  auto label = std::make_shared<abstract::AbstractTensor>(param.label_type, param.label_shape);
  auto weight = std::make_shared<abstract::AbstractTensor>(param.weight_type, param.weight_shape);
  auto total_weight = std::make_shared<abstract::AbstractTensor>(param.total_weight_type, param.total_weight_shape);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(label, nullptr);
  ASSERT_NE(weight, nullptr);
  ASSERT_NE(total_weight, nullptr);
  auto prim = std::make_shared<Primitive>(kNameNLLLossGrad_);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  if (param.is_success) {
    DoFuncImplInferAndCompare<NLLLossGradFuncImpl>(kNameNLLLossGrad_, {x, dy, label, weight, total_weight}, expect_shape, expect_type);
  } else {
    ASSERT_ANY_THROW(DoFuncImplInferAndCompare<NLLLossGradFuncImpl>(kNameNLLLossGrad_, {x, dy, label, weight, total_weight}, expect_shape, expect_type));
  }
}

INSTANTIATE_TEST_CASE_P(
  TestNLLLossGradGroup, TestNLLLossGrad,
  testing::Values(
    NLLLossGradOpParams{{2, 3}, kFloat32, {2, 3}, kFloat32, {2}, kInt32, {3}, kFloat32, {}, kFloat32, true,
                        {2, 3}, kFloat32},
    NLLLossGradOpParams{{-1, -1}, kFloat32, {-1, -1}, kFloat32, {-1}, kInt32, {-1}, kFloat32, {}, kFloat32, true,
                        {-1, -1}, kFloat32},
    NLLLossGradOpParams{{-2}, kFloat32, {-2}, kFloat32, {-2}, kInt32, {-2}, kFloat32, {}, kFloat32, true,
                        {-1, -1}, kFloat32},
    NLLLossGradOpParams{{2, 3}, kFloat32, {2, 3}, kFloat32, {3}, kInt32, {3}, kFloat32, {}, kFloat32, false},
    NLLLossGradOpParams{{2, 3}, kFloat32, {2, 3}, kFloat32, {2}, kInt32, {2}, kFloat32, {}, kFloat32, false}
 ));
}  // namespace ops
}  // namespace mindspore
