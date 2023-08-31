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
#include "ops/test_ops_cmp_utils.h"
#include "ops/ops_func_impl/angle.h"

namespace mindspore {
namespace ops {
static auto cases = testing::Values(
  EltwiseOpParams{{2, 3}, kComplex64, {2, 3}, kFloat32}, EltwiseOpParams{{2, -1}, kComplex64, {2, -1}, kFloat32},
  EltwiseOpParams{{-1, -1}, kComplex64, {-1, -1}, kFloat32}, EltwiseOpParams{{-2}, kComplex64, {-2}, kFloat32});

OP_FUNC_IMPL_TEST_DECLARE(Angle, EltwiseOpParams);

OP_FUNC_IMPL_TEST_CASES(Angle, cases);
}  // namespace ops
}  // namespace mindspore
