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

#include "plugin/device/cpu/kernel/pyboost/customize/non_zero.h"

#include "ir/scalar.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/pyboost/op_runner.h"
#include "kernel/pyboost/customize/op_common.h"
#include "mindspore/core/ops/auto_generate/gen_ops_primitive.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr NonZeroCPUCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "Nonzero CPU start";
  MS_EXCEPTION_IF_NULL(op);
  OpRunner::InferOpOutput(op, input_tensor);
  auto device_context = op->device_context();
  // Create device address for input tensors
  PyBoostUtils::PrepareOpInputs(device_context, op->stream_id(), input_tensor);
  // Create device address for output tensors
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  runtime::OpExecutor::GetInstance().WaitAll();

  // exec aclnnNonzero
  const auto &outputs = op->outputs();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(device_context, input_tensor);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);

  // Get inputs kernel tensors, the not-tensor value will malloc here
  const auto &input_address_info =
    PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), input_tensor);

  // Get outputs kernel tensors
  const auto &output_address_info =
    PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

  PyBoostUtils::LaunchKernel(op->primitive(), device_context, input_address_info, output_address_info);
  // update shape
  auto output_tensor_kernel = output_address_info.first;
  auto output_real_shape = output_tensor_kernel[0]->GetDeviceShapeVector();
  auto simple_infer_ptr = op->output_value_simple_info();
  simple_infer_ptr->shape_vector_ = ShapeArray{output_real_shape};
  op->UpdateOutputShape(outputs[kIndex0], output_real_shape);
  MS_LOG(DEBUG) << "NonZero CPU end";

  return outputs[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
