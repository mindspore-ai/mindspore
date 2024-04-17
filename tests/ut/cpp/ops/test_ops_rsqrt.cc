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
#include "ops/ops_func_impl/rsqrt.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_dyn_cases.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
class TestRsqrt : public TestOps,
                  public testing::WithParamInterface<std::tuple<EltwiseOpShapeParams, EltwiseOpTypeParams>> {};

TEST_P(TestRsqrt, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  auto expect_type = std::make_shared<TensorType>(dtype_param.out_type);
  DoFuncImplInferAndCompare<RsqrtFuncImpl>(kNameRsqrt, {x}, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(TestRsqrtGroup, TestRsqrt,
                        testing::Combine(EltwiseDynShapeTestCases,
                                         testing::ValuesIn({EltwiseOpTypeParams{kFloat16, kFloat16},
                                                            EltwiseOpTypeParams{kFloat32, kFloat32},
                                                            EltwiseOpTypeParams{kFloat64, kFloat64},
                                                            EltwiseOpTypeParams{kComplex64, kComplex64},
                                                            EltwiseOpTypeParams{kComplex128, kComplex128}})));
}  // namespace ops
}  // namespace mindspore
