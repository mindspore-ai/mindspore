/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/dequant_swiglu_quant.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr DequantSwigluQuantAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &x, const std::optional<TensorPtr> &weight_scale,
  const std::optional<TensorPtr> &activation_scale, const std::optional<TensorPtr> &bias,
  const std::optional<TensorPtr> &quant_scale, const std::optional<TensorPtr> &quant_offset,
  const std::optional<TensorPtr> &group_index, const BoolImmPtr &activate_left, const Int64ImmPtr &quant_mode) {
  MS_LOG(DEBUG) << "Call DequantSwigluQuant start";
  auto quant_mode_str =
    device::ascend::DequantSwigluQuantInputQuantMode::ConvertEnumToString(GetValue<int64_t>(quant_mode));
  auto activate_left_bool = GetValue<bool>(activate_left);
  OpRunner::InferOpOutput(op, x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index,
                          activate_left, quant_mode);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x, weight_scale, activation_scale, bias,
                                quant_scale, quant_offset, group_index);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, x, weight_scale, activation_scale, bias, quant_scale,
                                                  quant_offset, group_index, activate_left_bool, quant_mode_str]() {
      MS_LOG(DEBUG) << "Run device task DequantSwigluQuant start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x, weight_scale, activation_scale, bias, quant_scale, quant_offset,
                                   group_index);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnDequantSwigluQuant, device_context, op->stream_id(), x, weight_scale, activation_scale, bias,
                   quant_scale, quant_offset, group_index, activate_left_bool, quant_mode_str, outputs[0], outputs[1]);
      MS_LOG(DEBUG) << "Run device task DequantSwigluQuant end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
