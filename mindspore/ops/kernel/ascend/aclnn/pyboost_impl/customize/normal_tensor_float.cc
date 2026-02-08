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

#include "kernel/ascend/aclnn/pyboost_impl/customize/normal_tensor_float.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "utils/value_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr NormalTensorFloatAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &mean_tensor,
                                                   const ScalarPtr &std, const TensorPtr &seed,
                                                   const TensorPtr &offset) {
  MS_LOG(DEBUG) << "NormalTensorFloat call start";
  OpRunner::InferOpOutput(op, mean_tensor, std, seed, offset);
  auto std_imm = GetScalarCastValue<float>("NormalTensorFloat", std);
  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), mean_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, mean_tensor, std_imm, seed_imm, offset_imm]() {
      MS_LOG(DEBUG) << "Run device task NormalTensorFloat end";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, mean_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnNormalTensorFloat, device_context, op->stream_id(), mean_tensor, std_imm, seed_imm, offset_imm,
                   outputs[0]);
      MS_LOG(DEBUG) << "Run device task NormalTensorFloat end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
