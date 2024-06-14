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

#include "plugin/device/ascend/kernel/pyboost/customize/add_rmsnorm_quant_v2.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void AddRmsNormQuantAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x1_tensor,
                                    const BaseTensorPtr &x2_tensor, const BaseTensorPtr &gamma_tensor,
                                    const BaseTensorPtr &scale_tensor, const BaseTensorPtr &offset_tensor,
                                    const FP32ImmPtr &epsilon) {
  MS_LOG(DEBUG) << "Call start";
  OpRunner::InferOpOutput(op, x1_tensor, x2_tensor, gamma_tensor, scale_tensor, offset_tensor, epsilon);
  // Convert ValuePtr to c++ scalar
  auto epsilon_imm = static_cast<double>(GetValue<float>(epsilon));

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x1_tensor, x2_tensor, gamma_tensor, scale_tensor,
                                offset_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x1_tensor, x2_tensor, gamma_tensor, scale_tensor, offset_tensor, epsilon_imm]() {
      MS_LOG(DEBUG) << "Run device task AddRmsNormQuant start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x1_tensor, x2_tensor, gamma_tensor, scale_tensor, offset_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnAddRmsNormQuant, device_context, op->stream_id(), x1_tensor, x2_tensor, gamma_tensor,
                   scale_tensor, nullptr, offset_tensor, nullptr, nullptr, epsilon_imm, outputs[0], outputs[1],
                   outputs[2]);
      MS_LOG(DEBUG) << "Run device task AddRmsNormQuant end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
