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
#include "ops/ops_func_impl/logical_xor.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
OP_FUNC_IMPL_TEST_DECLARE(LogicalXor, MultiInputOpParams);

OP_FUNC_IMPL_TEST_CASES(
  LogicalXor,
  testing::Values(
    MultiInputOpParams{{{9}, {}}, {kBool, kBool}, {{9}}, {kBool}},
    MultiInputOpParams{{{2, 3}, {3}}, {kBool, kBool}, {{2, 3}}, {kBool}},
    MultiInputOpParams{{{2, 3}, {2, 1}}, {kBool, kBool}, {{2, 3}}, {kBool}},
    MultiInputOpParams{{{2, -1}, {2, 1}}, {kBool, kBool}, {{2, -1}}, {kBool}},
    MultiInputOpParams{{{2, 3}, {2, -1}}, {kBool, kBool}, {{2, 3}}, {kBool}},
    MultiInputOpParams{{{2, -1}, {2, -1}}, {kBool, kBool}, {{2, -1}}, {kBool}},
    MultiInputOpParams{{{2, 3}, {-1, 1}}, {kBool, kBool}, {{2, 3}}, {kBool}},
    MultiInputOpParams{{{-1, 3}, {2, 1}}, {kBool, kBool}, {{2, 3}}, {kBool}},
    MultiInputOpParams{{{2, 3}, {-2}}, {kBool, kBool}, {{-2}}, {kBool}},
    MultiInputOpParams{{{-2}, {2, 1}}, {kBool, kBool}, {{-2}}, {kBool}},
    MultiInputOpParams{{{-2}, {-2}}, {kBool, kBool}, {{-2}}, {kBool}}
  ));
}  // namespace ops
}  // namespace mindspore
