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

#include "plugin/device/ascend/kernel/pyboost/customize/sigmoid_grad.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void SigmoidGradAscendCall(const device::DeviceContext *device_context, const TensorPtr &dy_tensor,
                           const TensorPtr &y_tensor, const std::vector<tensor::TensorPtr> &outputs) {
  auto stream_ptr = device_context->device_res_manager_->GetStream(kDefaultStreamIndex);
  LAUNCH_ACLNN(aclnnSigmoidBackward, device_context, stream_ptr, dy_tensor, y_tensor, outputs[0]);
}
}  // namespace

tensor::TensorPtr SigmoidGradAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &y_tensor,
                                             const TensorPtr &dy_tensor) {
  OpRunner::InferOpOutput(op, dy_tensor, y_tensor);

  // Create device address for input/output tensors
  PyBoostUtils::PrepareOpInputs(op->device_context(), dy_tensor, y_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<pynative::PyBoostDeviceTask>([op, dy_tensor, y_tensor]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, dy_tensor, y_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
    SigmoidGradAscendCall(device_context, dy_tensor, y_tensor, outputs);
    MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
