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

#include "plugin/device/ascend/kernel/pyboost/customize/quant_batch_matmul.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void QuantMatmulV3AscendCall(const std::shared_ptr<OpRunner> &op, const device::DeviceContext *device_context,
                             const BaseTensorPtr &x1_tensor, const BaseTensorPtr &x2_tensor,
                             const BaseTensorPtr &scale_tensor, const std::optional<BaseTensorPtr> &offset_tensor,
                             const std::optional<BaseTensorPtr> &bias_tensor, const bool &transpose_x1,
                             const bool &transpose_x2, const std::vector<tensor::BaseTensorPtr> &outputs) {
  MS_LOG(DEBUG) << "QuantMatmulV3 call start";
  LAUNCH_ACLNN(aclnnQuantMatmulV3, device_context, op->stream_id(), x1_tensor, x2_tensor, scale_tensor, offset_tensor,
               bias_tensor, transpose_x1, transpose_x2, outputs[0]);
  MS_LOG(DEBUG) << "QuantMatmulV3 call end";
}
}  // namespace

tensor::BaseTensorPtr QuantMatmulV3AscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x1_tensor,
                                                   const BaseTensorPtr &x2_tensor, const BaseTensorPtr &scale_tensor,
                                                   const std::optional<BaseTensorPtr> &offset_tensor,
                                                   const std::optional<BaseTensorPtr> &bias_tensor,
                                                   const BoolImmPtr &transpose_x1, const BoolImmPtr &transpose_x2,
                                                   const Int64ImmPtr &dtype) {
  OpRunner::InferOpOutput(op, x1_tensor, x2_tensor, scale_tensor, offset_tensor, bias_tensor, transpose_x1,
                          transpose_x2, dtype);

  auto transpose_x1_imm = GetValue<bool>(transpose_x1);
  auto transpose_x2_imm = GetValue<bool>(transpose_x2);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x1_tensor, x2_tensor, scale_tensor,
                                offset_tensor, bias_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x1_tensor, x2_tensor, scale_tensor, offset_tensor, bias_tensor, transpose_x1_imm, transpose_x2_imm]() {
      MS_LOG(DEBUG) << "Run device task QuantMatmulV3 start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x1_tensor, x2_tensor, scale_tensor, offset_tensor, bias_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      QuantMatmulV3AscendCall(op, device_context, x1_tensor, x2_tensor, scale_tensor, offset_tensor, bias_tensor,
                              transpose_x1_imm, transpose_x2_imm, outputs);
      MS_LOG(DEBUG) << "Run device task QuantMatmulV3 end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
