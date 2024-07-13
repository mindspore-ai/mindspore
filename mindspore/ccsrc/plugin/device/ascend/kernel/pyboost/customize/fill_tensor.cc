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

#include "plugin/device/ascend/kernel/pyboost/customize/fill_tensor.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr FillTensorAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &size,
                                                const BaseTensorPtr &fill_value,
                                                const std::optional<Int64ImmPtr> &dtype) {
  OpRunner::InferOpOutput(op, size, fill_value, dtype);
  // No need to convert input
  bool is_host_tensor = fill_value->device_address() == nullptr && fill_value->isa<Tensor>();
  if (!is_host_tensor) {
    PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), fill_value);
  }
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, fill_value, is_host_tensor]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    if (is_host_tensor) {
      MS_LOG(INFO) << "For " << op->primitive()->name()
                   << ", Input [fill_value] is a host tensor, FillScalar will be used.";
      auto value = CreateValueFromTensor(fill_value->cast<TensorPtr>())->cast<ScalarPtr>();
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << "Call FillScalar start";
      LAUNCH_ACLNN(aclnnInplaceFillScalar, device_context, op->stream_id(), outputs[0], value);
      MS_LOG(DEBUG) << "Launch FillScalar end";
      return;
    }
    // Malloc for output tensors
    PyBoostUtils::MallocOpInputs(device_context, fill_value);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    MS_LOG(DEBUG) << "Call FillTensor start";
    LAUNCH_ACLNN(aclnnInplaceFillTensor, device_context, op->stream_id(), outputs[0], fill_value);
    MS_LOG(DEBUG) << "Launch FillTensor end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
