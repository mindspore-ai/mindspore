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

#include "plugin/device/ascend/kernel/pyboost/customize/masked_fill.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {

tensor::TensorPtr MaskedFillAscendCall(const std::shared_ptr<OpRunner> &op, const device::DeviceContext *device_context,
                                       const TensorPtr &input_tensor, const TensorPtr &mask_tensor,
                                       const TensorPtr &value_tensor, const TensorPtr &output_tensor) {
  LAUNCH_ACLNN(aclnnInplaceCopy, device_context, op->stream_id(), output_tensor, input_tensor);
  LAUNCH_ACLNN(aclnnInplaceMaskedFillTensor, device_context, op->stream_id(), output_tensor, mask_tensor, value_tensor);
  return output_tensor;
}
}  // namespace

tensor::TensorPtr MaskedFillAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                            const TensorPtr &mask_tensor, const TensorPtr &value_tensor,
                                            OpRunnerInfo *op_runner_info) {
  if (op_runner_info != nullptr) {
    OpRunner::InferOpOutput(op, op_runner_info);
  } else {
    OpRunner::InferOpOutput(op, input_tensor, mask_tensor, value_tensor);
  }
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, mask_tensor, value_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, mask_tensor, value_tensor]() {
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), input_tensor, mask_tensor, value_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), op->outputs());
      MaskedFillAscendCall(op, device_context, input_tensor, mask_tensor, value_tensor, op->output(0));
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
