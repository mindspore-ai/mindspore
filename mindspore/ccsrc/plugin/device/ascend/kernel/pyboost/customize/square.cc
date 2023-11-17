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

#include "plugin/device/ascend/kernel/pyboost/customize/square.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void SquareAscendCall(const device::DeviceContext *device_context, const tensor::TensorPtr &input_tensor,
                      const std::vector<tensor::TensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  auto stream_ptr = device_context->device_res_manager_->GetStream(kDefaultStreamIndex);
  constexpr int64_t val = 2;
  const auto exponent = std::dynamic_pointer_cast<Scalar>(MakeValue(val));
  MS_EXCEPTION_IF_NULL(exponent);
  LAUNCH_ACLNN(aclnnPowTensorScalar, device_context, stream_ptr, input_tensor, exponent, outputs[0]);
  MS_LOG(DEBUG) << "Launch end";
}
}  // namespace

tensor::TensorPtr SquareAscendCustomize(const std::shared_ptr<Op> &op, const TensorPtr &x_tensor) {
  Op::InferOpOutput(op, x_tensor);
  // No need to convert input
  // Async
  DispatchRun(std::make_shared<pynative::PyBoostDeviceTask>([op, x_tensor]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    runtime::DeviceAddressUtils::CreateInputAddress(device_context, x_tensor, "x_tensor");
    // Malloc for output tensors
    PrepareOpOutputs(device_context, outputs, op->device_sync_promises());
    SquareAscendCall(device_context, x_tensor, outputs);
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
