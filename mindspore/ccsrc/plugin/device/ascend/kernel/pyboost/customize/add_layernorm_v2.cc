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

#include "plugin/device/ascend/kernel/pyboost/customize/add_layernorm_v2.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void AddLayerNormAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x1_tensor,
                                 const BaseTensorPtr &x2_tensor, const BaseTensorPtr &gamma_tensor,
                                 const BaseTensorPtr &beta_tensor, const FP32ImmPtr &epsilon,
                                 const BoolImmPtr &additional_out) {
  MS_LOG(DEBUG) << "Call start";
  OpRunner::InferOpOutput(op, x1_tensor, x2_tensor, gamma_tensor, beta_tensor, epsilon, additional_out);
  // Convert ValuePtr to c++ scalar
  auto epsilon_imm = static_cast<double>(GetValue<float>(epsilon));
  auto additional_out_imm = GetValue<bool>(additional_out);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x1_tensor, x2_tensor, gamma_tensor, beta_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x1_tensor, x2_tensor, gamma_tensor, beta_tensor, epsilon_imm, additional_out_imm]() {
      MS_LOG(DEBUG) << "Run device task AddLayerNorm start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x1_tensor, x2_tensor, gamma_tensor, beta_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnAddLayerNorm, device_context, op->stream_id(), x1_tensor, x2_tensor, gamma_tensor, beta_tensor,
                   nullptr, epsilon_imm, additional_out_imm, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2],
                   outputs[kIndex3]);
      MS_LOG(DEBUG) << "Run device task AddLayerNorm end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
