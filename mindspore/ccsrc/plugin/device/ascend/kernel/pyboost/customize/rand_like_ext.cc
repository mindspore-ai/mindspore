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

#include "plugin/device/ascend/kernel/pyboost/customize/rand_like_ext.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr RandLikeExtAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                 const BaseTensorPtr &tensor_tensor,
                                                 const std::optional<Int64ImmPtr> &dtype, const Int64ImmPtr &seed,
                                                 const Int64ImmPtr &offset) {
  OpRunner::InferOpOutput(op, tensor_tensor, dtype, seed, offset);
  // ValueTuple to std::vector

  // Convert ValuePtr to c++ scalar
  // Convert ValuePtr to c++ scalar
  auto seed_imm = GetValue<int64_t>(seed);
  auto offset_imm = GetValue<int64_t>(offset);

  auto device_context = op->device_context();
  auto outputs = op->outputs();
  PyBoostUtils::PrepareOpInputs(device_context, op->stream_id(), tensor_tensor);

  PyBoostUtils::PrepareOpOutputs(device_context, op->stream_id(), outputs);

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, tensor_tensor, seed_imm, offset_imm]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, tensor_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    LAUNCH_ACLNN(aclnnInplaceUniform, device_context, op->stream_id(), outputs[0], 0., 1., seed_imm, offset_imm);
  }));
  return outputs[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
