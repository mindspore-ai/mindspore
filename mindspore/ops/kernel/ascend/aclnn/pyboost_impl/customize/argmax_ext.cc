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

#include "kernel/ascend/aclnn/pyboost_impl/customize/argmax_ext.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ArgMaxAscendCustomize(const TensorPtr &input_x_tensor, const std::optional<Int64ImmPtr> &dim,
                                        const BoolImmPtr &keepdim) {
  auto output = InferOutput<1>(prim::kPrimArgMaxExt, input_x_tensor, dim, keepdim);
  int64_t dim_imm = 0;
  bool keepdim_imm = GetValue<bool>(keepdim);

  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  auto input_x_imm = reshape(input_x_tensor, {-1});

  if (dim.has_value()) {
    dim_imm = GetValue<int64_t>(dim.value());
    input_x_imm = input_x_tensor;
  }
  // No need to convert input
  TensorToDevice(input_x_imm);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnArgMax, input_x_imm, dim_imm, keepdim_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
