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

#include "kernel/ascend/aclnn/pyboost_impl/customize/mse_loss_grad_ext.h"
#include <unordered_map>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "mindapi/base/types.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr MSELossGradExtAscendCustomize(const TensorPtr &grad_output_tensor, const TensorPtr &input_tensor,
                                                const TensorPtr &target_tensor, const Int64ImmPtr &reduction) {
  MS_LOG(DEBUG) << "MSELossGradExt call start";
  auto output = InferOutput<1>(prim::kPrimMSELossGradExt, grad_output_tensor, input_tensor, target_tensor, reduction);

  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  // transform reduction enum value to corresponding value
  auto reduction_value = ops::ConvertReductionForAclnn(reduction_imm);

  const std::vector<int64_t> &expand_shape = output->shape();

  TensorPtr input_tensor_bd = input_tensor;
  TensorPtr target_tensor_bd = target_tensor;

  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  const std::vector<int64_t> &input_shape = input_tensor->shape();
  if (input_shape != expand_shape) {
    input_tensor_bd = broadcast_to(input_tensor, expand_shape);
  }
  const std::vector<int64_t> &target_shape = target_tensor->shape();
  if (target_shape != expand_shape) {
    target_tensor_bd = broadcast_to(target_tensor, expand_shape);
  }

  // No need to convert input
  TensorToDevice(grad_output_tensor, input_tensor_bd, target_tensor_bd);
  TensorMalloc(output);

  // Async
  DISPATCH_ACLNN(aclnnMseLossBackward, grad_output_tensor, input_tensor_bd, target_tensor_bd, reduction_value, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
