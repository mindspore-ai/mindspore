/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/pyboost/customize/reduce_all.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {

tensor::BaseTensorPtr ReduceAllAscendCall(const std::shared_ptr<OpRunner> &op,
                                          const device::DeviceContext *device_context,
                                          const BaseTensorPtr &input_tensor, const std::vector<int64_t> &axis,
                                          const bool &keep_dims, const std::vector<tensor::BaseTensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  LAUNCH_ACLNN(aclnnAll, device_context, op->stream_id(), input_tensor, axis, keep_dims, outputs[0]);
  MS_LOG(DEBUG) << "Launch end";
  return outputs[0];
}
}  // namespace

tensor::BaseTensorPtr ReduceAllAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                               const std::optional<ValueTuplePtr> &axis, const BoolImmPtr &keep_dims) {
  OpRunner::InferOpOutput(op, input_tensor, axis, keep_dims);

  std::vector<int64_t> axis_vector{};
  // If axis is not None, Convert ValueTuple to std::vector
  if (axis.has_value()) {
    axis_vector = ConvertValueTupleToVector<int64_t>(axis.value());
  }

  // Convert ValuePtr to c++ scalar
  bool keep_dims_imm = GetValue<bool>(keep_dims);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, axis_vector, keep_dims_imm]() {
      MS_LOG(DEBUG) << "Run device task ReduceAll start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), input_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), op->outputs());
      ReduceAllAscendCall(op, device_context, input_tensor, axis_vector, keep_dims_imm, outputs);
      MS_LOG(DEBUG) << "Run device task ReduceAll end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
