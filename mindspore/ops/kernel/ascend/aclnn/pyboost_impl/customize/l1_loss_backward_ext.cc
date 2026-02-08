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

#include "kernel/ascend/aclnn/pyboost_impl/customize/l1_loss_backward_ext.h"
#include <memory>
#include <unordered_map>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindapi/base/types.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr L1LossBackwardExtAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                   const TensorPtr &grad_output_tensor, const TensorPtr &input_tensor,
                                                   const TensorPtr &target_tensor, const Int64ImmPtr &reduction) {
  MS_LOG(DEBUG) << "L1LossBackwardExt call start";
  OpRunner::InferOpOutput(op, grad_output_tensor, input_tensor, target_tensor, reduction);

  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  // transform reduction enum value to corresponding value
  auto reduction_value = ops::ConvertReductionForAclnn(reduction_imm);

  const auto &input_shape = input_tensor->shape();
  const auto &target_shape = target_tensor->shape();
  const auto &expand_shape = op->output_value_simple_info()->shape_vector_[kIndex0];
  const auto &expand_shape_ptr = expand_shape;

  auto expand_input_tensor = input_tensor;
  auto expand_target_tensor = target_tensor;

  if (input_shape != expand_shape) {
    expand_input_tensor = broadcast_to(input_tensor, expand_shape_ptr);
  }
  if (target_shape != expand_shape) {
    expand_target_tensor = broadcast_to(target_tensor, expand_shape_ptr);
  }

  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), grad_output_tensor, expand_input_tensor,
                                expand_target_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, grad_output_tensor, expand_input_tensor, expand_target_tensor, reduction_value]() {
      MS_LOG(DEBUG) << "Run device task L1LossBackwardExt start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, grad_output_tensor, expand_input_tensor, expand_target_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnL1LossBackward, device_context, op->stream_id(), grad_output_tensor, expand_input_tensor,
                   expand_target_tensor, reduction_value, outputs[0]);
      MS_LOG(DEBUG) << "Run device task L1LossBackwardExt end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
