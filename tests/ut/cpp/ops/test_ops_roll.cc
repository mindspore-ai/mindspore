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
#include "ops/ops_func_impl/roll.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_name.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct RollParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr shift;
  ValuePtr axis;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestRoll : public TestOps, public testing::WithParamInterface<RollParams> {};

TEST_P(TestRoll, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto expect = std::make_shared<abstract::AbstractTensor>(param.out_type, param.out_shape);
  ASSERT_NE(x, nullptr);
  auto prim = std::make_shared<Primitive>(kNameRoll);
  prim->AddAttr(kShift, MakeValue(param.shift));
  prim->AddAttr(kAxis, MakeValue(param.axis));

  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<RollFuncImpl>(kNameReLU, {x}, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestRollGroup, TestRoll,
  testing::Values(RollParams{{2, 3}, kFloat32, CreateTuple({1}), CreateTuple({1}), {2, 3}, kFloat32},
                  RollParams{{-1, 2, 3}, kFloat16, CreateTuple({0}), CreateTuple({0}), {-1, 2, 3}, kFloat16},
                  RollParams{{-1, -1}, kInt8, CreateTuple({1}), CreateTuple({1}), {-1, -1}, kInt8},
                  RollParams{{-2}, kUInt64, CreateTuple({1}), CreateTuple({1}), {-2}, kUInt64}));
}  // namespace ops
}  // namespace mindspore
