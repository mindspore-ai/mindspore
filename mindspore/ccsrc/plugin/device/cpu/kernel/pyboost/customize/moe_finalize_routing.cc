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
 * limitations under the License.plugin/device/cpu/hal/device
 */

#include "plugin/device/cpu/kernel/pyboost/customize/moe_finalize_routing.h"
#include <memory>
#include <functional>
#include "kernel/pyboost/pyboost_utils.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void MoeFinalizeRoutingCPUCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &expanded_x_tensor,
                                    const BaseTensorPtr &x1_tensor, const std::optional<BaseTensorPtr> &x2_tensor,
                                    const std::optional<BaseTensorPtr> &bias_tensor,
                                    const std::optional<BaseTensorPtr> &scales_tensor,
                                    const std::optional<BaseTensorPtr> &expanded_row_idx_tensor,
                                    const std::optional<BaseTensorPtr> &expaned_expert_idx_tensor) {
  MS_LOG(DEBUG) << "Call MoeFinalizeRouting start";

  OpRunner::InferOpOutput(op, expanded_x_tensor, x1_tensor, x2_tensor, bias_tensor, scales_tensor,
                          expanded_row_idx_tensor, expaned_expert_idx_tensor);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), expanded_x_tensor, x1_tensor, x2_tensor,
                                bias_tensor, scales_tensor, expanded_row_idx_tensor, expaned_expert_idx_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, expanded_x_tensor, x1_tensor, x2_tensor, bias_tensor,
                                                  scales_tensor, expanded_row_idx_tensor, expaned_expert_idx_tensor]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, expanded_x_tensor, x1_tensor, x2_tensor, bias_tensor, scales_tensor,
                                   expanded_row_idx_tensor, expaned_expert_idx_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      const auto &input_address_info = PyBoostUtils::GetAddressInfo(
        device_context, op->stream_id(), op->input_abs(), expanded_x_tensor, x1_tensor, x2_tensor, bias_tensor,
        scales_tensor, expanded_row_idx_tensor, expaned_expert_idx_tensor);
      const auto &output_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

      PyBoostUtils::LaunchKernel(op->primitive(), op->device_context(), input_address_info, output_address_info);
      MS_LOG(DEBUG) << "Launch MoeFinalizeRouting end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
