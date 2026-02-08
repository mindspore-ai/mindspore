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

#include "kernel/ascend/aclnn/pyboost_impl/customize/masked_fill_scalar.h"
#include <memory>
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
tensor::TensorPtr MaskedFillScalarAscendCall(const std::shared_ptr<OpRunner> &op,
                                             const device::DeviceContext *device_context, const TensorPtr &input_tensor,
                                             const TensorPtr &mask_tensor, const ScalarPtr &value,
                                             const TensorPtr &output_tensor) {
  LAUNCH_ACLNN(aclnnInplaceCopy, device_context, op->stream_id(), output_tensor, input_tensor);
  LAUNCH_ACLNN(aclnnInplaceMaskedFillScalar, device_context, op->stream_id(), output_tensor, mask_tensor, value);
  return output_tensor;
}
}  // namespace

tensor::TensorPtr MaskedFillScalarAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                                  const TensorPtr &mask_tensor, const ScalarPtr &value) {
  OpRunner::InferOpOutput(op, input_tensor, mask_tensor, value);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, mask_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, mask_tensor, value]() {
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor, mask_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, op->outputs());
    MaskedFillScalarAscendCall(op, device_context, input_tensor, mask_tensor, value, op->output(0));
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
