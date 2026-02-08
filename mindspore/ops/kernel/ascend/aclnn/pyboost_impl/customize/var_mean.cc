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

#include "kernel/ascend/aclnn/pyboost_impl/customize/var_mean.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void VarMeanAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                            const std::optional<ValueTuplePtr> &dim, const Int64ImmPtr &correction,
                            const BoolImmPtr &keepdim) {
  OpRunner::InferOpOutput(op, input_tensor, dim, correction, keepdim);

  std::vector<int64_t> dim_vector{};
  if (dim.has_value()) {
    dim_vector = ConvertValueTupleToVector<int64_t>(dim.value());
  }
  const auto correction_imm = GetValue<int64_t>(correction);
  const auto keepdim_imm = GetValue<bool>(keepdim);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, dim_vector, correction_imm, keepdim_imm]() {
      MS_LOG(DEBUG) << "Run device task VarMean start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, op->outputs());

      LAUNCH_ACLNN(aclnnVarMean, device_context, op->stream_id(), input_tensor, dim_vector, correction_imm, keepdim_imm,
                   outputs[kIndex0], outputs[kIndex1]);
      MS_LOG(DEBUG) << "Run device task VarMean end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
