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

#include "plugin/device/cpu/kernel/pyboost/customize/layer_norm_ext.h"
#include <memory>
#include <functional>
#include "kernel/pyboost/pyboost_utils.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void LayerNormExtCPUCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                              const ValueTuplePtr &normalized_shape, const std::optional<BaseTensorPtr> &weight_tensor,
                              const std::optional<BaseTensorPtr> &bias_opt_tensor, const FP32ImmPtr &eps) {
  MS_LOG(DEBUG) << "Call start";

  OpRunner::InferOpOutput(op, input_tensor, normalized_shape, weight_tensor, bias_opt_tensor, eps);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, weight_tensor, bias_opt_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, input_tensor, normalized_shape, weight_tensor, bias_opt_tensor, eps]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, weight_tensor, bias_opt_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      const auto &input_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), input_tensor, normalized_shape,
                                     weight_tensor, bias_opt_tensor, eps);
      const auto &output_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

      PyBoostUtils::LaunchKernel(op->primitive(), op->device_context(), input_address_info, output_address_info);
      MS_LOG(DEBUG) << "Launch end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
