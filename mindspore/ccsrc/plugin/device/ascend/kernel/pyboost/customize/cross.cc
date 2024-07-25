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

#include "plugin/device/ascend/kernel/pyboost/customize/cross.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "ops/ops_func_impl/cross.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::BaseTensorPtr CrossAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                           const BaseTensorPtr &other_tensor, const Int64ImmPtr &dim) {
  MS_LOG(DEBUG) << op->primitive()->name() << " call start";
  OpRunner::InferOpOutput(op, input_tensor, other_tensor, dim);
  auto dim_imm = GetValue<int64_t>(dim);
  const int64_t default_dim = -65530;
  if (dim_imm == default_dim) {
    const auto &input_shape = input_tensor->shape();
    const auto &other_shape = other_tensor->shape();
    dim_imm = SizeToLong(ops::CalCrossDimFromDefaultValue(input_shape, other_shape));
  }
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, other_tensor);

  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, other_tensor, dim_imm]() {
    MS_LOG(DEBUG) << "Run device task " << op->primitive()->name() << " end";
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostDeviceTask,
                                       op->primitive()->name(), false);
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor, other_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    LAUNCH_ACLNN(aclnnLinalgCross, device_context, op->stream_id(), input_tensor, other_tensor, dim_imm, outputs[0]);
    MS_LOG(DEBUG) << "Run device task " << op->primitive()->name() << " end";
  }));
  return op->outputs()[0];
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
