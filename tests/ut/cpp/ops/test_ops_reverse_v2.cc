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
#include "ops/ops_func_impl/reverse_v2.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct ReverseV2Params {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr axis;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestReverseV2 : public TestOps, public testing::WithParamInterface<ReverseV2Params> {};

TEST_P(TestReverseV2, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto axis = param.axis->ToAbstract();
  ASSERT_NE(axis, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<ReverseV2FuncImpl>(kNameReverseV2, {x, axis}, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(TestReverseV2Group, TestReverseV2,
                        testing::Values(ReverseV2Params{{2, 3}, kFloat32, CreateTuple({1}), {2, 3}, kFloat32},
                                        ReverseV2Params{{-1, 2, 3}, kFloat16, CreateTuple({0}), {-1, 2, 3}, kFloat16},
                                        ReverseV2Params{{-1, -1}, kInt8, CreateTuple({1}), {-1, -1}, kInt8},
                                        ReverseV2Params{{-2}, kUInt64, CreateTuple({1}), {-2}, kUInt64}));
}  // namespace ops
}  // namespace mindspore
