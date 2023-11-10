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

#include "common/common_test.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/ops_func_impl/masked_fill.h"

namespace mindspore::ops {
OP_FUNC_IMPL_TEST_DECLARE(MaskedFill, MultiInputOpParams);

OP_FUNC_IMPL_TEST_CASES(
  MaskedFill,
  testing::Values(MultiInputOpParams{{{4, 3, 3}, {3, 3}, {}}, {kUInt8, kBool, kUInt8}, {{4, 3, 3}}, {kUInt8}, {}},
                  MultiInputOpParams{{{4, -1, -1}, {3, 3}, {}}, {kInt16, kBool, kInt16}, {{4, 3, 3}}, {kInt16}, {}},
                  MultiInputOpParams{{{4, 3, 3}, {-1, -1}, {}}, {kInt32, kBool, kInt32}, {{4, 3, 3}}, {kInt32}, {}},
                  MultiInputOpParams{{{-1, -1}, {-1, -1}, {}}, {kFloat32, kBool, kFloat32}, {{-1, -1}}, {kFloat32}, {}},
                  MultiInputOpParams{{{-2}, {-2}, {}}, {kFloat64, kBool, kFloat64}, {{-2}}, {kFloat64}, {}},
                  MultiInputOpParams{{{-2}, {1}, {}}, {kComplex128, kBool, kComplex128}, {{-2}}, {kComplex128}, {}}));
}  // namespace mindspore::ops
