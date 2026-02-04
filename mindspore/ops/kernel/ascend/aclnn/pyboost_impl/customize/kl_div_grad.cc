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

#include "kernel/ascend/aclnn/pyboost_impl/customize/kl_div_grad.h"
#include <memory>
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr KLDivGradAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &grad_output_tensor,
                                           const TensorPtr &input_tensor, const TensorPtr &target_tensor,
                                           const Int64ImmPtr &reduction, const BoolImmPtr &log_target) {
  OpRunner::InferOpOutput(op, grad_output_tensor, input_tensor, target_tensor, reduction, log_target);

  auto log_target_imm = GetValue<bool>(log_target);
  auto reduction_enum = static_cast<Reduction>(GetValue<int64_t>(reduction));
  // transform reduction enum value to corresponding value
  auto reduction_imm = ops::ConvertReductionForAclnn(reduction_enum);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), grad_output_tensor, input_tensor, target_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, grad_output_tensor, input_tensor, target_tensor, reduction_imm, log_target_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();

      PyBoostUtils::MallocOpInputs(device_context, grad_output_tensor, input_tensor, target_tensor);
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      LAUNCH_ACLNN(aclnnKlDivBackward, device_context, op->stream_id(), grad_output_tensor, input_tensor, target_tensor,
                   reduction_imm, log_target_imm, outputs[0]);
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
