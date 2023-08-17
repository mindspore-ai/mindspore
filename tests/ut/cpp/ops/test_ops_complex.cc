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
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/complex.h"
#include "ops/test_ops_dyn_cases.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
class TestComplex : public TestOps, public testing::WithParamInterface<BroadcastOpParams> {};

TEST_P(TestComplex, complex_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("Complex");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto y = std::make_shared<abstract::AbstractTensor>(param.y_type, param.y_shape);
  ASSERT_NE(x, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(x), std::move(y)};
  auto infer_impl = std::make_shared<ComplexFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = infer_impl->InferShape(primitive, input_args);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = infer_impl->InferType(primitive, input_args);
  ASSERT_NE(infer_type, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  ASSERT_NE(expect_type, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  ASSERT_TRUE(*infer_type == *expect_type);
}

INSTANTIATE_TEST_CASE_P(TestComplexGroup, TestComplex,
                        testing::Values(BroadcastOpParams{{-1, -1}, kFloat32, {1, 1}, kFloat32, {-1, -1}, kComplex64},
                                        BroadcastOpParams{{-1, -1}, kFloat32, {2, 3}, kFloat32, {2, 3}, kComplex64},
                                        BroadcastOpParams{{-2}, kFloat32, {2, 3}, kFloat32, {-2}, kComplex64},
                                        BroadcastOpParams{{-1, -1}, kFloat64, {1, 1}, kFloat64, {-1, -1}, kComplex128},
                                        BroadcastOpParams{{-1, -1}, kFloat64, {2, 3}, kFloat64, {2, 3}, kComplex128},
                                        BroadcastOpParams{{-2}, kFloat64, {2, 3}, kFloat64, {-2}, kComplex128}));
}  // namespace ops
}  // namespace mindspore
