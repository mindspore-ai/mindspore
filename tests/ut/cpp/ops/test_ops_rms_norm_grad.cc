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
#include "ops/ops_func_impl/rms_norm_grad.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct TestRmsNormGradParams {
  TypePtr dtype;
  ShapeVector dy_shape;
  ShapeVector x_shape;
  ShapeVector rstd_shape;
  ShapeVector gamma_shape;
  ShapeVector dx_shape;
  ShapeVector dgamma_shape;
};

class TestRmsNormGrad : public TestOps, public testing::WithParamInterface<TestRmsNormGradParams> {};

TEST_P(TestRmsNormGrad, rms_norm_dyn_grad_shape) {
  const auto &param = GetParam();
  auto dy = std::make_shared<abstract::AbstractTensor>(param.dtype, param.dy_shape);
  auto x = std::make_shared<abstract::AbstractTensor>(param.dtype, param.x_shape);
  auto rstd = std::make_shared<abstract::AbstractTensor>(param.dtype, param.rstd_shape);
  auto gamma = std::make_shared<abstract::AbstractTensor>(param.dtype, param.gamma_shape);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(rstd, nullptr);
  ASSERT_NE(gamma, nullptr);
  auto infer_impl = std::make_shared<RmsNormGradFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(dy), std::move(x), std::move(rstd), std::move(gamma)};
  auto dx_shape = std::make_shared<abstract::Shape>(param.dx_shape);
  auto dx_type = std::make_shared<TensorType>(param.dtype);
  auto dgamma_shape = std::make_shared<abstract::Shape>(param.dgamma_shape);

  auto expect_shape = std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{dx_shape, dgamma_shape});
  auto expect_type = std::make_shared<Tuple>(std::vector<TypePtr>{dx_type, std::make_shared<TensorType>(kFloat32)});
  DoFuncImplInferAndCompare<RmsNormGradFuncImpl>(kNameRmsNormGrad, input_args, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(TestRmsNormGrad, TestRmsNormGrad,
    testing::Values(TestRmsNormGradParams{kFloat32, {-2}, {-2}, {-2}, {-1, -1, -1}, {-2}, {-1, -1, -1}},
                    TestRmsNormGradParams{kFloat16, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-2}, {-1, -1, -1}, {-2}},
                    TestRmsNormGradParams{kFloat32, {2, 3, 4}, {2, 3, 4}, {2, 1, 1}, {2, 3}, {2, 3, 4}, {2, 3}}));
}  // namespace ops
}  // namespace mindspore
