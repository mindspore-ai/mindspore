/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/pyboost/ops/cast.h"
#include <memory>
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr CastCPU::Call(const TensorPtr &input_tensor, const TypePtr &type) {
  MS_LOG(DEBUG) << "Call start";

  InferOutput(input_tensor, type);

  PyBoostUtils::PrepareOpInputs(device_context_, input_tensor);
  PyBoostUtils::PrepareOpOutputs(device_context_, outputs_);

  // Async
  auto op = get_op();
  PyBoostUtils::DispatchRun(std::make_shared<pynative::PyBoostDeviceTask>([this, op, input_tensor, type]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();

    PyBoostUtils::MallocOpInputs(device_context, input_tensor);
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    const auto &input_address_info = PyBoostUtils::GetAddressInfo(device_context, op->input_abs(), input_tensor);
    const auto &output_address_info = PyBoostUtils::GetAddressInfo(device_context, {op->output_abs()}, outputs);

    PyBoostUtils::LaunchKernel(primitive(), op->device_context(), input_address_info, output_address_info);
  }));
  MS_LOG(DEBUG) << "Call end";
  return outputs_[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
