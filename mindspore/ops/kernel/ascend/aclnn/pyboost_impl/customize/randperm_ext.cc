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

#include "kernel/ascend/aclnn/pyboost_impl/customize/randperm_ext.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr RandpermExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const Int64ImmPtr &n,
                                             const TensorPtr &seed, const TensorPtr &offset,
                                             const std::optional<Int64ImmPtr> &dtype) {
  OpRunner::InferOpOutput(op, n, seed, offset, dtype);
  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);
  auto n_imm = GetValueWithCheck<int64_t>(n);
  auto device_context = op->device_context();
  auto outputs = op->outputs();
  PyBoostUtils::PrepareOpOutputs(device_context, op->stream_id(), outputs);

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, n_imm, seed_imm, offset_imm]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    LAUNCH_ACLNN(aclnnRandperm, device_context, op->stream_id(), n_imm, seed_imm, offset_imm, outputs[kIndex0]);
  }));
  return outputs[kIndex0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
