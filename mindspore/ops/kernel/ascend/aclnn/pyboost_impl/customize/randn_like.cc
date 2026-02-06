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

#include "kernel/ascend/aclnn/pyboost_impl/customize/randn_like.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr RandnLikeAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &tensor_tensor,
                                           const TensorPtr &seed, const TensorPtr &offset,
                                           const std::optional<Int64ImmPtr> &dtype,
                                           const std::optional<Int64ImmPtr> &device) {
  OpRunner::InferOpOutput(op, tensor_tensor, seed, offset, dtype, device);

  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);
  auto device_context = op->device_context();
  auto outputs = op->outputs();
  PyBoostUtils::PrepareOpInputs(device_context, op->stream_id(), tensor_tensor);

  PyBoostUtils::PrepareOpOutputs(device_context, op->stream_id(), outputs);

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, tensor_tensor, seed_imm, offset_imm]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    constexpr float mean_imm = 0.0;
    constexpr float std_imm = 1.0;
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, tensor_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    LAUNCH_ACLNN(aclnnInplaceNormal, device_context, op->stream_id(), outputs[kIndex0], mean_imm, std_imm, seed_imm,
                 offset_imm);
  }));
  return outputs[kIndex0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
