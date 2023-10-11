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

#include "ir/dtype/number.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/ops_func_impl/assign.h"

namespace mindspore::ops {
OP_FUNC_IMPL_TEST_DECLARE(Assign, MultiInputOpParams);

OP_FUNC_IMPL_TEST_CASES(
  Assign,
  testing::Values(
    MultiInputOpParams{{{1}, {}}, {kFloat16, kFloat16}, {{1}}, {kFloat16}, {}},
    MultiInputOpParams{{{}, {1}}, {kFloat32, kFloat32}, {{}}, {kFloat32}, {}},
    MultiInputOpParams{{{2, 3, 4}, {2, 3, 4}}, {kFloat64, kFloat64}, {{2, 3, 4}}, {kFloat64}, {}},
    MultiInputOpParams{{{1, 2}, {-2}}, {kUInt16, kUInt16}, {{-2}}, {kUInt16}, {}},
    MultiInputOpParams{{{-1, -1}, {1, 2}}, {kBool, kBool}, {{-1, -1}}, {kBool}, {}},
    MultiInputOpParams{{{-2}, {-1}}, {kComplex128, kComplex128}, {{-2}}, {kComplex128}, {}}
  ));
}  // namespace mindspore::ops
