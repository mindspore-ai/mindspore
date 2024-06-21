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

#include "plugin/device/ascend/kernel/pyboost/customize/normal_float_float.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr NormalFloatFloatAscendCustomize(const std::shared_ptr<OpRunner> &op, const FP32ImmPtr &mean_float,
                                                      const FP32ImmPtr &std_float, const ValueTuplePtr &size,
                                                      const Int64ImmPtr &seed, const Int64ImmPtr &offset) {
  MS_LOG(DEBUG) << "NormalFloatFloat call start";
  OpRunner::InferOpOutput(op, mean_float, std_float, size, seed, offset);
  // ValueTuple to std::vector
  auto mean_imm = GetValue<float>(mean_float);
  auto std_imm = GetValue<float>(std_float);
  auto seed_imm = static_cast<int64_t>(GetValue<int64_t>(seed));
  auto offset_imm = static_cast<int64_t>(GetValue<int64_t>(offset));

  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, mean_imm, std_imm, seed_imm, offset_imm]() {
      MS_LOG(DEBUG) << "Run device task NormalFloatFloat end";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnNormalFloatFloat, device_context, op->stream_id(), mean_imm, std_imm, seed_imm, offset_imm,
                   outputs[0]);
      MS_LOG(DEBUG) << "Run device task NormalFloatFloat end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
