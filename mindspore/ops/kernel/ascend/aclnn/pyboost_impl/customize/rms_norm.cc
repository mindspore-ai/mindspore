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

#include "kernel/ascend/aclnn/pyboost_impl/customize/rms_norm.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> RmsNormAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                                        const TensorPtr &x_tensor,
                                                                        const TensorPtr &gamma_tensor,
                                                                        const FP32ImmPtr &epsilon) {
  OpRunner::InferOpOutput(op, x_tensor, gamma_tensor, epsilon);
  auto epsilon_imm = static_cast<double>(GetValue<float>(epsilon));

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor, gamma_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor, gamma_tensor, epsilon_imm]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, x_tensor, gamma_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
    LAUNCH_ACLNN(aclnnRmsNorm, device_context, op->stream_id(), x_tensor, gamma_tensor, epsilon_imm, outputs[0],
                 outputs[1]);
    MS_LOG(DEBUG) << op->primitive()->name() << " Call end";
  }));
  return std::make_tuple(op->output(0), op->output(1));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
