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
#include "ops/ops_func_impl/maximum_grad.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
OP_FUNC_IMPL_TEST_DECLARE(MaximumGrad, MultiInputOpParams);

OP_FUNC_IMPL_TEST_CASES(
  MaximumGrad,
  testing::Values(
    MultiInputOpParams{{{9}, {}, {9}}, {kFloat32, kFloat32, kFloat32}, {{9}, {}}, {kFloat32, kFloat32},
                       {CreateScalar<bool>(true), CreateScalar<bool>(true)}},
    MultiInputOpParams{{{2, 3}, {3}, {2, 3}}, {kFloat32, kFloat32, kFloat32}, {{2, 3}, {3}}, {kFloat32, kFloat32},
                       {CreateScalar<bool>(false), CreateScalar<bool>(false)}},
    MultiInputOpParams{{{2, 3}, {2, 1}, {2, 3}}, {kFloat32, kFloat32, kFloat32}, {{2, 3}, {2, 1}}, {kFloat32, kFloat32},
                       {CreateScalar(kValueAny), CreateScalar(kValueAny)}},
    MultiInputOpParams{{{2, -1}, {2, 1}, {2, -1}}, {kFloat32, kFloat32, kFloat32}, {{2, -1}, {2, 1}}, {kFloat32, kFloat32},
                       {CreateScalar(kValueAny), CreateScalar(kValueAny)}},
    MultiInputOpParams{{{2, 3}, {2, -1}, {2, 3}}, {kFloat32, kFloat32, kFloat32}, {{2, 3}, {2, -1}}, {kFloat32, kFloat32},
                       {CreateScalar(kValueAny), CreateScalar(kValueAny)}},
    MultiInputOpParams{{{2, -1}, {2, -1}, {2, -1}}, {kFloat32, kFloat32, kFloat32}, {{2, -1}, {2, -1}}, {kFloat32, kFloat32},
                       {CreateScalar(kValueAny), CreateScalar(kValueAny)}},
    MultiInputOpParams{{{2, 3}, {-1, 1}, {2, 3}}, {kFloat32, kFloat32, kFloat32}, {{2, 3}, {-1, 1}}, {kFloat32, kFloat32},
                       {CreateScalar(kValueAny), CreateScalar(kValueAny)}},
    MultiInputOpParams{{{-1, 3}, {2, 1}, {2, 3}}, {kFloat32, kFloat32, kFloat32}, {{-1, 3}, {2, 1}}, {kFloat32, kFloat32},
                       {CreateScalar(kValueAny), CreateScalar(kValueAny)}},
    MultiInputOpParams{{{2, 3}, {-2}, {-2}}, {kFloat32, kFloat32, kFloat32}, {{2, 3}, {-2}}, {kFloat32, kFloat32},
                       {CreateScalar(kValueAny), CreateScalar(kValueAny)}},
    MultiInputOpParams{{{-2}, {2, 1}, {-2}}, {kFloat32, kFloat32, kFloat32}, {{-2}, {2, 1}}, {kFloat32, kFloat32},
                       {CreateScalar(kValueAny), CreateScalar(kValueAny)}},
    MultiInputOpParams{{{-2}, {-2}, {-2}}, {kFloat32, kFloat32, kFloat32}, {{-2}, {-2}}, {kFloat32, kFloat32},
                       {CreateScalar(kValueAny), CreateScalar(kValueAny)}}
  ));
}  // namespace ops
}  // namespace mindspore
