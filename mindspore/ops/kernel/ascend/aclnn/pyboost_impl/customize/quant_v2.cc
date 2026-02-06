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

#include "kernel/ascend/aclnn/pyboost_impl/customize/quant_v2.h"
#include <string>
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"

namespace mindspore {
using mindspore::device::ascend::AscendQuantRoundMode;
namespace kernel {
namespace pyboost {
void QuantV2AscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor,
                            const TensorPtr &scale_tensor, const TensorPtr &offset_tensor, const BoolImmPtr &sqrt_mode,
                            const Int64ImmPtr &rounding_mode, const Int64ImmPtr &dst_type) {
  OpRunner::InferOpOutput(op, x_tensor, scale_tensor, offset_tensor, sqrt_mode, rounding_mode, dst_type);

  // Convert ValuePtr to c++ scalar
  auto sqrt_mode_imm = GetValue<bool>(sqrt_mode);
  auto rounding_mode_imm = GetValue<int64_t>(rounding_mode);
  std::string rounding_mode_str = AscendQuantRoundMode::ConvertEnumToString(rounding_mode_imm);

  // Infer function has confirmed the actual dtype of output
  TypeId out_type = op->output_abs()->GetType()->cast<TensorTypePtr>()->element()->type_id();

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor, scale_tensor, offset_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x_tensor, scale_tensor, offset_tensor, sqrt_mode_imm, rounding_mode_str, out_type]() {
      MS_LOG(DEBUG) << "Run device task QuantV2 start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x_tensor, scale_tensor, offset_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnAscendQuant, device_context, op->stream_id(), x_tensor, scale_tensor, offset_tensor,
                   sqrt_mode_imm, rounding_mode_str, out_type, outputs[0]);
      MS_LOG(DEBUG) << "Run device task QuantV2 end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
