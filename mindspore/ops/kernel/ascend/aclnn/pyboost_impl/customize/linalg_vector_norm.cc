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

#include "kernel/ascend/aclnn/pyboost_impl/customize/linalg_vector_norm.h"
#include <memory>
#include <functional>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void LinalgVectorNormAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor,
                                     const FP32ImmPtr &ord, const std::optional<ValueTuplePtr> &dim,
                                     const BoolImmPtr &keepdim, const std::optional<Int64ImmPtr> &dtype) {
  MS_LOG(DEBUG) << "Call LinalgVectorNorm start";
  // Convert ValuePtr to c++ scalar
  OpRunner::InferOpOutput(op, x_tensor, ord, dim, keepdim, dtype);
  std::vector<int64_t> dim_vector{};
  if (dim.has_value()) {
    dim_vector = ConvertValueTupleToVector<int64_t>(dim.value());
  }
  ScalarPtr ord_scalar = nullptr;
  MAKE_SCALAR(GetValue<float>(ord), kNumberTypeFloat32, ord_scalar);
  const auto keepdim_imm = GetValue<bool>(keepdim);

  TypeId out_dtype = dtype.has_value() ? static_cast<TypeId>(GetValue<int64_t>(dtype.value())) : x_tensor->data_type();
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor, ord_scalar, dim_vector, keepdim_imm, out_dtype]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnLinalgVectorNorm, device_context, op->stream_id(), x_tensor, ord_scalar, dim_vector,
                   keepdim_imm, out_dtype, outputs[kIndex0]);
      MS_LOG(DEBUG) << "Launch LinalgVectorNorm end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
