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
#include "plugin/device/ascend/kernel/pyboost/customize/isclose.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr IsCloseAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                             const BaseTensorPtr &other_tensor, const FP32ImmPtr &rtol,
                                             const FP32ImmPtr &atol, const BoolImmPtr &equal_nan) {
  MS_LOG(DEBUG) << "IsCloseCustomize start";
  OpRunner::InferOpOutput(op, input_tensor, other_tensor, rtol, atol, equal_nan);

  auto rtol_imm = static_cast<double>(GetValue<float>(rtol));
  auto atol_imm = static_cast<double>(GetValue<float>(atol));
  auto equal_nan_imm = GetValue<bool>(equal_nan);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, other_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, other_tensor, rtol_imm, atol_imm, equal_nan_imm]() {
      MS_LOG(DEBUG) << "Run device task IsClose start";
      runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostDeviceTask,
                                         "IsClose", false);
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, other_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnIsClose, device_context, op->stream_id(), input_tensor, other_tensor, rtol_imm, atol_imm,
                   equal_nan_imm, outputs[0]);
      MS_LOG(DEBUG) << "Run device task IsClose end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
