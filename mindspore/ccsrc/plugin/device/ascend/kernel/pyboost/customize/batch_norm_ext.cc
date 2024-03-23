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

#include "plugin/device/ascend/kernel/pyboost/customize/batch_norm_ext.h"
#include <memory>
#include <functional>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void BatchNormExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                 const BaseTensorPtr &weight_tensor, const BaseTensorPtr &bias_tensor,
                                 const BaseTensorPtr &mean_tensor, const BaseTensorPtr &variance_tensor,
                                 const BoolImmPtr &training, const FP32ImmPtr &momentum, const FP32ImmPtr &epsilon) {
  MS_LOG(DEBUG) << "Call aclnnBatchNorm start";
  // Convert ValuePtr to c++ scalar
  OpRunner::InferOpOutput(op, input_tensor, weight_tensor, bias_tensor, mean_tensor, variance_tensor, training,
                          momentum, epsilon);
  auto training_imm = GetValue<bool>(training);
  auto momentum_imm = static_cast<double>(GetValue<float>(momentum));
  auto eps_imm = static_cast<double>(GetValue<float>(epsilon));

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, weight_tensor, bias_tensor,
                                mean_tensor, variance_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, weight_tensor, bias_tensor, mean_tensor,
                                                  variance_tensor, training_imm, momentum_imm, eps_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, weight_tensor, bias_tensor, mean_tensor,
                                   variance_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnBatchNorm, device_context, op->stream_id(), input_tensor, weight_tensor, bias_tensor,
                   mean_tensor, variance_tensor, training_imm, momentum_imm, eps_imm, outputs[kIndex0],
                   outputs[kIndex1], outputs[kIndex2]);
      op->set_outputs({op->output(0), op->output(kIndex1), op->output(kIndex2)});
      MS_LOG(DEBUG) << "Launch aclnnBatchNorm end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
