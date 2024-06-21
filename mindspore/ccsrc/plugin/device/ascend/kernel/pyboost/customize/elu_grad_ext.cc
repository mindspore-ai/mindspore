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

#include "plugin/device/ascend/kernel/pyboost/customize/elu_grad_ext.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr EluGradExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &dy_tensor,
                                                const BaseTensorPtr &x_tensor, const ScalarPtr &alpha) {
  OpRunner::InferOpOutput(op, dy_tensor, x_tensor, alpha);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), dy_tensor, x_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, dy_tensor, x_tensor, alpha]() {
    MS_LOG(DEBUG) << "Run device task EluGradExt start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, dy_tensor, x_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    // Convert ValuePtr to c++ scalar
    static const ScalarPtr scale = std::make_shared<FP32Imm>(1.f);
    static const ScalarPtr input_scale = std::make_shared<FP32Imm>(1.f);

    LAUNCH_ACLNN(aclnnEluBackward, device_context, op->stream_id(), dy_tensor, alpha, scale, input_scale, false,
                 x_tensor, outputs[0]);
    MS_LOG(DEBUG) << "Run device task EluGradExt end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
