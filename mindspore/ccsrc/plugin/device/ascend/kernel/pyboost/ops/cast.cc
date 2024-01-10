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

#include "plugin/device/ascend/kernel/pyboost/ops/cast.h"

#include <memory>
#include <vector>
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr CastAscend::Call(const TensorPtr &input_tensor, const TypePtr &type) {
  MS_LOG(DEBUG) << "Call start";
  InferOutput(input_tensor, type);
  // ValueTuple to std::vector

  // Convert ValuePtr to c++ scalar

  PyBoostUtils::PrepareOpInputs(device_context_, input_tensor);
  PyBoostUtils::PrepareOpOutputs(device_context_, outputs_);

  // Async
  auto op = get_op();
  PyBoostUtils::DispatchRun(std::make_shared<pynative::PyBoostDeviceTask>([op, input_tensor, type]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(op->device_context(), input_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(op->device_context(), op->outputs());

    auto stream_ptr = device::ascend::AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
    LAUNCH_ACLNN(aclnnCast, device_context, stream_ptr, input_tensor, type, outputs[0]);
    MS_LOG(DEBUG) << "Launch end";
  }));
  return outputs_[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
