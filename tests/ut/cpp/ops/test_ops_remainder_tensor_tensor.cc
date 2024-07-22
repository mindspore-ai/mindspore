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
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "ops/ops_func_impl/remainder_tensor_tensor.h"

namespace mindspore {
namespace ops {
OP_FUNC_IMPL_INFER_TEST_DECLARE(RemainderTensorTensor, MultiInputOpParams);
OP_FUNC_IMPL_INFER_TEST_CASES(
  RemainderTensorTensor,
  testing::Values(
    MultiInputOpParams{{{1, 3}, {2, 1}}, {kFloat32, kFloat32}, {{2, 3}}, {kFloat32}, {}},
    MultiInputOpParams{{{-1, 3}, {-1, 1}}, {kFloat32, kFloat64}, {{-1, 3}}, {kFloat64}, {}},
    MultiInputOpParams{{{-2}, {2, 3}}, {kFloat16, kBFloat16}, {{-2}}, {kFloat32}, {}},
    MultiInputOpParams{{{-1, 1, 3}, {1, -1, 3}}, {kInt64, kFloat32}, {{-1, -1, 3}}, {kFloat32}, {}},
    MultiInputOpParams{{{-1, 2, 3}, {2, -1, 3}}, {kBFloat16, kInt32}, {{2, 2, 3}}, {kBFloat16}, {}},
    MultiInputOpParams{{{2, 1, 4, 5, 6, 9}, {9}}, {kFloat16, kInt32}, {{2, 1, 4, 5, 6, 9}}, {kFloat16}, {}},
    MultiInputOpParams{{{2, 1, 4, -1}, {-1, -1, 4, 5}}, {kFloat32, kInt32}, {{2, -1, 4, 5}}, {kFloat32}, {}},
    MultiInputOpParams{{{-2}, {2, 1, 4, 5, -1, 9}}, {kBFloat16, kInt64}, {{-2}}, {kBFloat16}, {}}
  ));
}  // namespace ops
}  // namespace mindspore