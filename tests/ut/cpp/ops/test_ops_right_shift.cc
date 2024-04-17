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
#include "ops/ops_func_impl/right_shift.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
class TestRightShift : public TestOps, public testing::WithParamInterface<BroadcastOpParams> {};

TEST_P(TestRightShift, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto y = std::make_shared<abstract::AbstractTensor>(param.y_type, param.y_shape);
  ASSERT_NE(y, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<RightShiftFuncImpl>(kNameRightShift, {x, y}, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(TestRightShiftGroup, TestRightShift,
                        testing::Values(BroadcastOpParams{{1, 3}, kInt8, {2, 1}, kInt8, {2, 3}, kInt8},
                                        BroadcastOpParams{{-1, 3}, kInt16, {-1, 1}, kInt16, {-1, 3}, kInt16},
                                        BroadcastOpParams{{-1, 3}, kInt32, {-1, 1}, kInt32, {-1, 3}, kInt32},
                                        BroadcastOpParams{{-1, 1, 3}, kUInt8, {1, -1, 3}, kUInt8, {-1, -1, 3}, kUInt8},
                                        BroadcastOpParams{{-1, 2, 3}, kUInt16, {2, -1, 3}, kUInt16, {2, 2, 3}, kUInt16},
                                        BroadcastOpParams{{-2}, kUInt32, {2, 3}, kUInt32, {-2}, kUInt32}));
}  // namespace ops
}  // namespace mindspore
