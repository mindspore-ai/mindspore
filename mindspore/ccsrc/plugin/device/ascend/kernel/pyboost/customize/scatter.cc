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

#include "plugin/device/ascend/kernel/pyboost/customize/scatter.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr ScatterAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                             const Int64ImmPtr &dim, const BaseTensorPtr &index_tensor,
                                             const BaseTensorPtr &src_tensor, const Int64ImmPtr &reduce) {
  auto op_name = op->primitive()->name();
  MS_LOG(DEBUG) << op_name << " call start";
  OpRunner::InferOpOutput(op, input_tensor, dim, index_tensor, src_tensor, reduce);

  auto dim_imm = GetValue<int64_t>(dim);
  auto reduce_imm = GetValue<int64_t>(reduce);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, index_tensor, src_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, dim_imm, index_tensor, src_tensor, reduce_imm]() {
      auto op_name = op->primitive()->name();
      MS_LOG(DEBUG) << "Run device task " << op_name << " end";
      runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostDeviceTask,
                                         op_name, false);
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, index_tensor, src_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnInplaceCopy, device_context, op->stream_id(), outputs[0], input_tensor);
      LAUNCH_ACLNN(aclnnScatter, device_context, op->stream_id(), input_tensor, dim_imm, index_tensor, src_tensor,
                   reduce_imm, outputs[0]);
      MS_LOG(DEBUG) << "Run device task " << op_name << " end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
