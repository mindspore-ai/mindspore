/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/abs.h"

#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore::kernel::pyboost {
tensor::TensorPtr AbsAscendCustomize(const tensor::TensorPtr &input_tensor) {
  if (input_tensor->data_type() == kNumberTypeComplex128) {
    kernel::pyboost::RequireGradGuard require_grad_guard(false);
    auto real = real_view(input_tensor);
    auto imag = imag_view(input_tensor);
    return sqrt(add(square(real), square(imag)));
  }

  auto output = InferOutput<1>(prim::kPrimAbs, input_tensor);
  TensorToDevice(input_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnAbs, input_tensor, output);
  return output;
}
}  // namespace mindspore::kernel::pyboost
