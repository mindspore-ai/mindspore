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

#include "plugin/device/ascend/kernel/pyboost/customize/masked_select_grad.h"

#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::BaseTensorPtr MaskedSelectGradAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                      const BaseTensorPtr &input_tensor,
                                                      const BaseTensorPtr &mask_tensor,
                                                      const BaseTensorPtr &grad_tensor) {
  MS_LOG(DEBUG) << op->primitive()->name() << " call start";
  OpRunner::InferOpOutput(op, input_tensor, mask_tensor, grad_tensor);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, mask_tensor, grad_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, mask_tensor, grad_tensor]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, mask_tensor, grad_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnInplaceZero, device_context, op->stream_id(), outputs[0]);
      LAUNCH_ACLNN(aclnnInplaceMaskedScatter, device_context, op->stream_id(), outputs[0], mask_tensor, grad_tensor);
      MS_LOG(DEBUG) << "Run device task " << op->primitive()->name() << " end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
