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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_uniform.h"
#include <memory>
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
double GetScalarValue(const std::shared_ptr<Scalar> &scalar, const string &scalar_name) {
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
    MS_EXCEPTION(TypeError) << "For Pyboost InplaceUniform, the input " << scalar_name
                            << " does not support dtype: " << scalar->type_name();
  }
}
}  // namespace

tensor::TensorPtr InplaceUniformAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                                const ScalarPtr &from, const ScalarPtr &to, const TensorPtr &seed,
                                                const TensorPtr &offset) {
  MS_LOG(DEBUG) << "aclnnInplaceUniform call start";

  // Convert ValuePtr to c++ scalar
  double from_imm = GetScalarValue(from, "from");
  double to_imm = GetScalarValue(to, "to");

  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  op->set_outputs({input_tensor});

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, from_imm, to_imm, seed_imm, offset_imm]() {
      MS_LOG(DEBUG) << "Run device task aclnnInplaceUniform start";
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor);

      LAUNCH_ACLNN(aclnnInplaceUniform, device_context, op->stream_id(), input_tensor, from_imm, to_imm,
                   static_cast<uint64_t>(seed_imm), static_cast<uint64_t>(offset_imm));
      MS_LOG(DEBUG) << "Run device task aclnnInplaceUniform end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
