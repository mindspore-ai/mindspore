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
#include "ops/ops_func_impl/hswish_grad.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_dyn_cases.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
class TestHSwishGrad
    : public TestOps,
      public testing::WithParamInterface<std::tuple<EltwiseGradOpShapeParams, EltwiseGradOpTypeParams>> {};

TEST_P(TestHSwishGrad, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto grad = std::make_shared<abstract::AbstractTensor>(dtype_param.grad_type, shape_param.grad_shape);
  ASSERT_NE(grad, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  auto expect_type = std::make_shared<TensorType>(dtype_param.out_type);
  DoFuncImplInferAndCompare<HSwishGradFuncImpl>(kNameHSwishGrad, {grad, x}, expect_shape, expect_type);
}

namespace {
auto HSwishGradOpTypeCases = testing::ValuesIn({
  EltwiseGradOpTypeParams{kInt8, kInt8, kInt8},
  EltwiseGradOpTypeParams{kInt16, kInt16, kInt16},
  EltwiseGradOpTypeParams{kInt32, kInt32, kInt32},
  EltwiseGradOpTypeParams{kInt64, kInt64, kInt64},
  EltwiseGradOpTypeParams{kFloat16, kFloat16, kFloat16},
  EltwiseGradOpTypeParams{kFloat32, kFloat32, kFloat32},
  EltwiseGradOpTypeParams{kFloat64, kFloat64, kFloat64},
  EltwiseGradOpTypeParams{kBFloat16, kBFloat16, kBFloat16},
});
}

OP_FUNC_IMPL_SIMPLEINFER_TEST_DECLARE(HSwishGrad, MultiInputOpParams);
OP_FUNC_IMPL_SIMPLEINFER_TEST_CASES(
  HSwishGrad, testing::Values(MultiInputOpParams{{{2, 3}, {2, 3}}, {kInt8, kInt8}, {{2, 3}}, {kInt8}, {}},
                              MultiInputOpParams{{{2, 3}, {2, 3}}, {kInt16, kInt16}, {{2, 3}}, {kInt16}, {}},
                              MultiInputOpParams{{{2, 3}, {2, 3}}, {kInt32, kInt32}, {{2, 3}}, {kInt32}, {}},
                              MultiInputOpParams{{{2, 3}, {2, 3}}, {kInt64, kInt64}, {{2, 3}}, {kInt64}, {}},
                              MultiInputOpParams{{{2, 3}, {2, 3}}, {kFloat16, kFloat16}, {{2, 3}}, {kFloat16}, {}},
                              MultiInputOpParams{{{2, 3}, {2, 3}}, {kFloat32, kFloat32}, {{2, 3}}, {kFloat32}, {}},
                              MultiInputOpParams{{{2, 3}, {2, 3}}, {kFloat64, kFloat64}, {{2, 3}}, {kFloat64}, {}},
                              MultiInputOpParams{{{2, 3}, {2, 3}}, {kBFloat16, kBFloat16}, {{2, 3}}, {kBFloat16}, {}}));

INSTANTIATE_TEST_CASE_P(TestHSwishGradGroup, TestHSwishGrad,
                        testing::Combine(EltwiseGradDynShapeTestCases, HSwishGradOpTypeCases));
}  // namespace ops
}  // namespace mindspore
