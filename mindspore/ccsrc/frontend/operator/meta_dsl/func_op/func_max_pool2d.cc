/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/func_max_pool2d.h"
#include "ir/dtype/type.h"
#include "utils/shape_utils.h"
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore::prim {
BeginFunction(FuncMaxPool2D, input, kernel_size, stride, padding, dilation, ceil_mode, return_indices) {
  auto get_real_stride = [&]() -> NodePtr {
    auto true_branch = [&]() { Return(stride); };
    auto false_branch = [&]() { Return(kernel_size); };
    auto condition = IsNotNone(stride);
    return If(condition, true_branch, false_branch);
  };
  auto real_stride = get_real_stride();

  static const auto argmax_type = Value(static_cast<int64_t>(kNumberTypeInt64));

  auto return_indices_true_branch = [&]() {
    auto outputs =
      Call(Prim(MaxPoolWithIndices), input, kernel_size, real_stride, padding, dilation, ceil_mode, argmax_type);
    Return(outputs);
  };
  auto return_indices_false_branch = [&]() {
    auto outputs =
      Call(Prim(MaxPoolWithMask), input, kernel_size, real_stride, padding, dilation, ceil_mode, argmax_type);
    Return(outputs);
  };
  Return(If(return_indices, return_indices_true_branch, return_indices_false_branch));
}
EndFunction(FuncMaxPool2D)
}  // namespace mindspore::prim
