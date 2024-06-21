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
#include "ops/ops_func_impl/dropout_ext.h"
#include "ops/ops_func_impl/dropout_gen_mask_ext.h"
#include "ops/ops_func_impl/dropout_do_mask_ext.h"
#include "ops/ops_func_impl/dropout_grad_ext.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

// DropoutExt
OP_FUNC_IMPL_TEST_DECLARE(DropoutExt, MultiInputOpParams);
OP_FUNC_IMPL_TEST_CASES(
  DropoutExt,
  testing::Values(
    MultiInputOpParams{{{1280, 77, 77}}, {kFloat32}, {{1280, 77, 77}, {948640}}, {kFloat32, kUInt8},
        {kValueAny, kValueAny, kValueAny}},
    MultiInputOpParams{{{-1, 77, 77}}, {kFloat32}, {{-1, 77, 77}, {-1}}, {kFloat32, kUInt8},
        {kValueAny, kValueAny, kValueAny}},
    MultiInputOpParams{{{1280, -1, -1}}, {kFloat32}, {{1280, -1, -1}, {-1}}, {kFloat32, kUInt8},
        {kValueAny, kValueAny, kValueAny}},
    MultiInputOpParams{{{-2}}, {kFloat32}, {{-2}, {-1}}, {kFloat32, kUInt8},
        {kValueAny, kValueAny, kValueAny}}
  ));

// DropoutGenMaskExt
OP_FUNC_IMPL_TEST_DECLARE(DropoutGenMaskExt, MultiInputOpParams);
OP_FUNC_IMPL_TEST_CASES(
  DropoutGenMaskExt,
  testing::Values(
    MultiInputOpParams{{}, {}, {{948640}}, {kUInt8}, {CreatePyIntTuple({1280, 77, 77})}},
    MultiInputOpParams{{}, {}, {{-1}}, {kUInt8}, {CreatePyIntTuple({kValueAny, 77, 77})}},
    MultiInputOpParams{{}, {}, {{-1}}, {kUInt8}, {kValueAny}}
  ));

const auto test_do_mask_cases = testing::Values(
    MultiInputOpParams{{{1280, 77, 77}, {948640}}, {kFloat32, kUInt8}, {{1280, 77, 77}}, {kFloat32},
        {kValueAny}},
    MultiInputOpParams{{{-1, 77, 77}, {-2}}, {kFloat32, kUInt8}, {{-1, 77, 77}}, {kFloat32},
        {kValueAny}},
    MultiInputOpParams{{{1280, -1, -1}, {-2}}, {kFloat32, kUInt8}, {{1280, -1, -1}}, {kFloat32},
        {kValueAny}},
    MultiInputOpParams{{{-2}, {-2}}, {kFloat32, kUInt8}, {{-2}}, {kFloat32},
        {kValueAny}}
  );

// DropoutDoMaskExt
OP_FUNC_IMPL_TEST_DECLARE(DropoutDoMaskExt, MultiInputOpParams);
OP_FUNC_IMPL_TEST_CASES(
  DropoutDoMaskExt,
  test_do_mask_cases);

// DropoutGradExt
OP_FUNC_IMPL_TEST_DECLARE(DropoutGradExt, MultiInputOpParams);
OP_FUNC_IMPL_TEST_CASES(
  DropoutGradExt,
  test_do_mask_cases);

}  // namespace ops
}  // namespace mindspore