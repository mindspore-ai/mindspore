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
double GetScalarValue(const std::shared_ptr<Scalar> &scalar) {
  if (scalar->isa<BoolImm>()) {
    return GetValue<bool>(scalar);
  } else if (scalar->isa<Int32Imm>()) {
    return GetValue<int32_t>(scalar);
  } else if (scalar->isa<Int64Imm>()) {
    return GetValue<int64_t>(scalar);
  } else if (scalar->isa<FP32Imm>()) {
    return GetValue<float>(scalar);
  } else if (scalar->isa<FP64Imm>()) {
    return GetValue<double>(scalar);
  } else {
    MS_EXCEPTION(TypeError) << "Unsupported type: " << scalar->type_name();
  }
}

tensor::BaseTensorPtr UniformExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &tensor_tensor,
                                                const ScalarPtr &a, const ScalarPtr &b, const BaseTensorPtr &seed,
                                                const BaseTensorPtr &offset) {
  MS_LOG(DEBUG) << "UniformExt call start";
  OpRunner::InferOpOutput(op, tensor_tensor, a, b, seed, offset);
  // ValueTuple to std::vector

  // Convert ValuePtr to c++ scalar
  // Convert ValuePtr to c++ scalar
  double a_imm = GetScalarValue(a);
  double b_imm = GetScalarValue(b);

  auto [seed_imm, offset_imm] = UpdateGeneratorState(seed, offset);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), tensor_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, tensor_tensor, a_imm, b_imm, seed_imm, offset_imm]() {
      MS_LOG(DEBUG) << "Run device task UniformExt end";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, tensor_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnInplaceUniform, device_context, op->stream_id(), outputs[0], a_imm, b_imm,
                   static_cast<uint64_t>(seed_imm), static_cast<uint64_t>(offset_imm));
      MS_LOG(DEBUG) << "Run device task UniformExt end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
