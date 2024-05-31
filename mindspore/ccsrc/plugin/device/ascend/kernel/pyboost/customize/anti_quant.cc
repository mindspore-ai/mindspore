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

#include "plugin/device/ascend/kernel/pyboost/customize/anti_quant.h"
#include <memory>
#include <functional>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void AntiQuantAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor,
                              const BaseTensorPtr &scale_tensor, const std::optional<BaseTensorPtr> &offset_opt_tensor,
                              const BoolImmPtr &sqrt_mode, const Int64ImmPtr &dtype) {
  MS_LOG(DEBUG) << "Call AntiQuant start";
  // Convert ValuePtr to c++ scalar
  OpRunner::InferOpOutput(op, x_tensor, scale_tensor, offset_opt_tensor, sqrt_mode, dtype);

  TypeId dtype_imm = static_cast<TypeId>(GetValue<int64_t>(dtype));
  auto sqrt_mode_imm = GetValue<bool>(sqrt_mode);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor, scale_tensor, offset_opt_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x_tensor, scale_tensor, offset_opt_tensor, dtype_imm, sqrt_mode_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x_tensor, scale_tensor, offset_opt_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnAscendAntiQuant, device_context, op->stream_id(), x_tensor, scale_tensor, offset_opt_tensor,
                   dtype_imm, sqrt_mode_imm, outputs[kIndex0]);
      MS_LOG(DEBUG) << "Launch AntiQuant end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
