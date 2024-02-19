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

#include "plugin/device/ascend/kernel/pyboost/customize/argmax_ext.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "kernel/pyboost/auto_generate/reshape.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ArgMaxAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_x_tensor,
                                        const std::optional<Int64ImmPtr> &dim, const BoolImmPtr &keepdim) {
  int64_t dim_imm = 0;
  bool keepdim_imm = GetValue<bool>(keepdim);

  auto reshape_op = CREATE_PYBOOST_OP(Reshape, op->device_context()->device_context_key_.device_name_);
  auto input_x_imm = reshape_op->Call(
    input_x_tensor, std::make_shared<ValueTuple>(std::vector<ValuePtr>({std::make_shared<Int64Imm>(-1)})));

  if (dim.has_value()) {
    dim_imm = GetValue<int64_t>(dim.value());
    input_x_imm = input_x_tensor;
  }
  OpRunner::InferOpOutput(op, input_x_imm, dim, keepdim);

  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_x_imm);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_x_imm, dim_imm, keepdim_imm]() {
    MS_LOG(DEBUG) << "Run device task ArgMax end";
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostDeviceTask,
                                       "ArgMax", false);
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_x_imm);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    LAUNCH_ACLNN(aclnnArgMax, device_context, op->stream_id(), input_x_imm, dim_imm, keepdim_imm, outputs[0]);
    MS_LOG(DEBUG) << "Run device task ArgMax end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
