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

#include "plugin/device/ascend/kernel/pyboost/customize/uniform_ext.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr UniformExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &tensor_tensor,
                                                const FP32ImmPtr &a, const FP32ImmPtr &b, const Int64ImmPtr &seed,
                                                const Int64ImmPtr &offset) {
  MS_LOG(DEBUG) << "UniformExt call start";
  OpRunner::InferOpOutput(op, tensor_tensor, a, b, seed, offset);
  // ValueTuple to std::vector

  // Convert ValuePtr to c++ scalar
  // Convert ValuePtr to c++ scalar
  auto a_imm = static_cast<double>(GetValue<float>(a));
  auto b_imm = static_cast<double>(GetValue<float>(b));
  auto seed_imm = static_cast<uint64_t>(GetValue<int64_t>(seed));
  auto offset_imm = static_cast<uint64_t>(GetValue<int64_t>(offset));

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), tensor_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, tensor_tensor, a_imm, b_imm, seed_imm,
                                                                          offset_imm]() {
    MS_LOG(DEBUG) << "Run device task UniformExt end";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, tensor_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    LAUNCH_ACLNN(aclnnInplaceCopy, device_context, op->stream_id(), outputs[0], tensor_tensor);
    LAUNCH_ACLNN(aclnnInplaceUniform, device_context, op->stream_id(), outputs[0], a_imm, b_imm, seed_imm, offset_imm);
    MS_LOG(DEBUG) << "Run device task UniformExt end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
