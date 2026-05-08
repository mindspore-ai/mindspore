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

#include "kernel/ascend/aclnn/pyboost_impl/customize/outer.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "primitive/auto_generate/gen_ops_primitive_o.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr OuterAscendCustomize(const TensorPtr &input, const TensorPtr &vec2) {
  auto output = InferOutput<1>(prim::kPrimOuter, input, vec2);

  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  auto real_input = reshape(input, {-1, 1});

  TensorToDevice(real_input, vec2);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnMul, real_input, vec2, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
