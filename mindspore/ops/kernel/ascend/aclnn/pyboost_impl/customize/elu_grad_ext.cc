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

#include "kernel/ascend/aclnn/pyboost_impl/customize/elu_grad_ext.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr EluGradExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &dy_tensor,
                                            const TensorPtr &x_or_out_tensor, const ScalarPtr &alpha,
                                            const BoolImmPtr &is_result) {
  const auto is_result_imm = GetValue<bool>(is_result);
  OpRunner::InferOpOutput(op, dy_tensor, x_or_out_tensor, alpha, is_result);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), dy_tensor, x_or_out_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, dy_tensor, x_or_out_tensor, alpha, is_result_imm]() {
      MS_LOG(DEBUG) << "Run device task EluGradExt start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, dy_tensor, x_or_out_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      // Convert ValuePtr to c++ scalar
      static const ScalarPtr scale = std::make_shared<FP32Imm>(1.f);
      static const ScalarPtr input_scale = std::make_shared<FP32Imm>(1.f);

      LAUNCH_ACLNN(aclnnEluBackward, device_context, op->stream_id(), dy_tensor, alpha, scale, input_scale,
                   is_result_imm, x_or_out_tensor, outputs[0]);
      MS_LOG(DEBUG) << "Run device task EluGradExt end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
