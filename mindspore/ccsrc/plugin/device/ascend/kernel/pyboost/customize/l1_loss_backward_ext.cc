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

#include "plugin/device/ascend/kernel/pyboost/customize/l1_loss_backward_ext.h"
#include <memory>
#include <unordered_map>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "mindapi/base/types.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr L1LossBackwardExtAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                       const BaseTensorPtr &grad_output_tensor,
                                                       const BaseTensorPtr &input_tensor,
                                                       const BaseTensorPtr &target_tensor,
                                                       const Int64ImmPtr &reduction) {
  MS_LOG(DEBUG) << "L1LossBackwardExt call start";
  OpRunner::InferOpOutput(op, grad_output_tensor, input_tensor, target_tensor, reduction);

  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  // transform reduction enum value to corresponding value
  auto reduction_value = ConvertReductionForAclnn(reduction_imm);

  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), grad_output_tensor, input_tensor, target_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, grad_output_tensor, input_tensor, target_tensor, reduction_value]() {
      MS_LOG(DEBUG) << "Run device task L1LossBackwardExt start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, grad_output_tensor, input_tensor, target_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnL1LossBackward, device_context, op->stream_id(), grad_output_tensor, input_tensor,
                   target_tensor, reduction_value, outputs[0]);
      MS_LOG(DEBUG) << "Run device task L1LossBackwardExt end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
