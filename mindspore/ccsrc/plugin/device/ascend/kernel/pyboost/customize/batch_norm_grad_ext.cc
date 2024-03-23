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

#include "plugin/device/ascend/kernel/pyboost/customize/batch_norm_grad_ext.h"
#include <memory>
#include <functional>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void BatchNormGradExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &dout_tensor,
                                     const BaseTensorPtr &input_tensor, const BaseTensorPtr &weight_tensor,
                                     const BaseTensorPtr &running_mean_tensor, const BaseTensorPtr &runnning_var_tensor,
                                     const BaseTensorPtr &saved_mean_tensor, const BaseTensorPtr &saved_rstd_tensor,
                                     const BoolImmPtr &training, const FP32ImmPtr &eps) {
  MS_LOG(DEBUG) << "Call aclnnBatchNormBackward start";
  // Convert ValuePtr to c++ scalar
  OpRunner::InferOpOutput(op, dout_tensor, input_tensor, weight_tensor, running_mean_tensor, runnning_var_tensor,
                          saved_mean_tensor, saved_rstd_tensor, training, eps);
  auto training_imm = GetValue<bool>(training);
  auto eps_imm = static_cast<double>(GetValue<float>(eps));

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), dout_tensor, input_tensor, weight_tensor,
                                running_mean_tensor, runnning_var_tensor, saved_mean_tensor, saved_rstd_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, dout_tensor, input_tensor, weight_tensor, running_mean_tensor, runnning_var_tensor, saved_mean_tensor,
     saved_rstd_tensor, training_imm, eps_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      std::vector<uint8_t> output_mask{1, 1, 1};
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, dout_tensor, input_tensor, weight_tensor, running_mean_tensor,
                                   runnning_var_tensor, saved_mean_tensor, saved_rstd_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnBatchNormBackward, device_context, op->stream_id(), dout_tensor, input_tensor, weight_tensor,
                   running_mean_tensor, runnning_var_tensor, saved_mean_tensor, saved_rstd_tensor, training_imm,
                   eps_imm, output_mask, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
      MS_LOG(DEBUG) << "Launch aclnnBatchNormBackward end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
