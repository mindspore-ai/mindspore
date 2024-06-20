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
#include <memory>
#include "common/common_test.h"
#include "ops/ops_func_impl/elu_ext.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

class TestEluExt : public TestOps, public testing::WithParamInterface<EltwiseOpParams> {};

TEST_P(TestEluExt, dyn_shape) {
  const auto &param = GetParam();
  auto input_x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.out_type);
  auto alpha = CreateScalar(1.f)->ToAbstract();

  EluExtFuncImpl elu_func;
  auto prim = std::make_shared<Primitive>("EluExt");

  auto out_dtype = elu_func.InferType(prim, {input_x, alpha});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape = elu_func.InferShape(prim, {input_x, alpha});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

OP_FUNC_IMPL_TEST_CASES(EluExt, testing::Values(EltwiseOpParams{{2, 3}, kFloat32, {2, 3}, kFloat32},
                                                EltwiseOpParams{{2, -1}, kFloat32, {2, -1}, kFloat32},
                                                EltwiseOpParams{{-1, -1}, kFloat32, {-1, -1}, kFloat32},
                                                EltwiseOpParams{{-2}, kFloat32, {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
