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

#include <memory>
#include "common/common_test.h"
#include "ops/ops_func_impl/floor.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
OP_FUNC_IMPL_TEST_DECLARE(Floor, EltwiseOpParams);
INSTANTIATE_TEST_CASE_P(TestFloor, TestFloor,
                        testing::Values(EltwiseOpParams{{2, 3}, kFloat16, {2, 3}, kFloat16},
                                        EltwiseOpParams{{2, 3}, kFloat32, {2, 3}, kFloat32},
                                        EltwiseOpParams{{2, 3}, kFloat64, {2, 3}, kFloat64},
                                        EltwiseOpParams{{2, 3}, kBFloat16, {2, 3}, kBFloat16},
                                        EltwiseOpParams{{2, -1}, kFloat32, {2, -1}, kFloat32},
                                        EltwiseOpParams{{-1, -1}, kFloat32, {-1, -1}, kFloat32},
                                        EltwiseOpParams{{-2}, kFloat32, {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
