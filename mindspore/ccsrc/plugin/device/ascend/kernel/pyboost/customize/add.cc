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

#include "plugin/device/ascend/kernel/pyboost/customize/add.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr AddAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor,
                                     const TensorPtr &y_tensor) {
  OpRunner::InferOpOutput(op, x_tensor, y_tensor);
  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), x_tensor, y_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<pynative::PyBoostDeviceTask>([op, x_tensor, y_tensor]() {
    MS_LOG(DEBUG) << "Run device task Add start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, x_tensor, y_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    auto stream_ptr = device_context->device_res_manager_->GetStream(kDefaultStreamIndex);
    ScalarPtr alpha = std::make_shared<FP32Imm>(1);
    LAUNCH_ACLNN(aclnnAdd, device_context, stream_ptr, x_tensor, y_tensor, alpha, outputs[0]);
    MS_LOG(DEBUG) << "Run device task Add end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
