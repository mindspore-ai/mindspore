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
#include "ops/ops_func_impl/bool_not.h"
#include "ops/test_ops_dyn_cases.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
class TestBoolNot : public TestOps, public testing::WithParamInterface<EltwiseOpTypeParams> {};

TEST_P(TestBoolNot, bool_not_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("BoolNot");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractScalar>(param.x_type);
  ASSERT_NE(x, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(x)};
  auto infer_impl = std::make_shared<BoolNotFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = infer_impl->InferShape(primitive, input_args);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = infer_impl->InferType(primitive, input_args);
  ASSERT_NE(infer_type, nullptr);

  auto expect_shape = abstract::kNoShape;
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = param.out_type;
  ASSERT_NE(expect_type, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  ASSERT_TRUE(*infer_type == *expect_type);
};

INSTANTIATE_TEST_CASE_P(TestBoolNotGroup, TestBoolNot,
                        testing::Values(EltwiseOpTypeParams{kBool, kBool}, EltwiseOpTypeParams{kInt8, kBool},
                                        EltwiseOpTypeParams{kInt16, kBool}, EltwiseOpTypeParams{kInt32, kBool},
                                        EltwiseOpTypeParams{kInt64, kBool}, EltwiseOpTypeParams{kUInt8, kBool},
                                        EltwiseOpTypeParams{kUInt16, kBool}, EltwiseOpTypeParams{kUInt32, kBool},
                                        EltwiseOpTypeParams{kFloat16, kBool}, EltwiseOpTypeParams{kFloat32, kBool},
                                        EltwiseOpTypeParams{kFloat64, kBool}, EltwiseOpTypeParams{kComplex64, kBool},
                                        EltwiseOpTypeParams{kComplex128, kBool}));
}  // namespace ops
}  // namespace mindspore
