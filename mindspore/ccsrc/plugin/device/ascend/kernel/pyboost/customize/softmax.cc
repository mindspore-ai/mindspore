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

#include "plugin/device/ascend/kernel/pyboost/customize/softmax.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void SoftmaxAscendCall(const device::DeviceContext *device_context, const tensor::TensorPtr &logits_tensor,
                       const int64_t dim, const std::vector<tensor::TensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  auto stream_ptr = device_context->device_res_manager_->GetStream(kDefaultStreamIndex);
  LAUNCH_ACLNN(aclnnSoftmax, device_context, stream_ptr, logits_tensor, dim, outputs[0]);
  MS_LOG(DEBUG) << "Launch end";
}
}  // namespace

tensor::TensorPtr SoftmaxAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &logits_tensor,
                                         const ValueTuplePtr &axis) {
  OpRunner::InferOpOutput(op, logits_tensor, axis);

  // ValueTuple to std::vector
  auto axis_vector = ConvertValueTupleToVector<int64_t>(axis);
  auto dim = axis_vector[0];

  PyBoostUtils::PrepareOpInputs(op->device_context(), logits_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<pynative::PyBoostDeviceTask>([op, logits_tensor, dim]() {
    MS_LOG(DEBUG) << "Run device task Softmax start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, logits_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    // Call aclnnSoftmax
    SoftmaxAscendCall(device_context, logits_tensor, dim, outputs);
    MS_LOG(DEBUG) << "Run device task Softmax end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
