/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "ops/ops_func_impl/select.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
OP_FUNC_IMPL_TEST_DECLARE(Select, MultiInputOpParams);

OP_FUNC_IMPL_TEST_CASES(
  Select,
  testing::Values(
    MultiInputOpParams{{{2, 3}, {2, 3}, {2, 3}}, {kBool, kFloat32, kFloat32}, {{2, 3}}, {kFloat32}, {}},
    MultiInputOpParams{{{2, 3}, {2, 3}, {2, 3}}, {kBool, kFloat32, kInt32}, {{2, 3}}, {kFloat32}, {}},
    MultiInputOpParams{{{-1, 3}, {2, 3}, {2, 3}}, {kBool, kFloat32, kFloat32}, {{2, 3}}, {kFloat32}, {}},
    MultiInputOpParams{{{2, -1}, {2, 3}, {2, 3}}, {kBool, kFloat32, kFloat32}, {{2, 3}}, {kFloat32}, {}},
    MultiInputOpParams{{{2, -1}, {2, -1}, {2, -1}}, {kBool, kFloat32, kFloat32}, {{2, -1}}, {kFloat32}, {}},
    MultiInputOpParams{{{-1, -1}, {-1, -1}, {2, -1}}, {kBool, kFloat32, kFloat32}, {{2, -1}}, {kFloat32}, {}},
    MultiInputOpParams{{{-1, -1}, {-1, -1}, {-1, -1}}, {kBool, kFloat32, kFloat32}, {{-1, -1}}, {kFloat32}, {}},
    MultiInputOpParams{{{4, 5, 8}, {1, 5, 8}, {4, 1, 8}}, {kBool, kFloat32, kFloat32}, {{4, 5, 8}}, {kFloat32}, {}},
    MultiInputOpParams{{{1, 65, 54, 12, 5, 2}, {5, 5, 65, 1, 12, 5, 2}, {65, 54, 1, 5, 2}},
                       {kBool, kFloat32, kFloat32},
                       {{5, 5, 65, 54, 12, 5, 2}},
                       {kFloat32},
                       {}}));
}  // namespace ops
}  // namespace mindspore
