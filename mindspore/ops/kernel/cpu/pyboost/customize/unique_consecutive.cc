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

#include "mindspore/ops/kernel/cpu/pyboost/customize/unique_consecutive.h"

#include "ir/scalar.h"
#include "kernel/cpu/cpu_kernel.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "include/pynative/utils/pyboost/customize/op_common.h"
#include "include/runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> UniqueConsecutiveCPUCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, const BoolImmPtr &return_inverse,
  const BoolImmPtr &return_counts, const std::optional<Int64ImmPtr> &dim) {
  MS_LOG(WARNING) << "Run device task unique_consecutive start";
  auto device_context = op->device_context();
  auto stream_id = op->stream_id();
  OpRunner::InferOpOutput(op, input_tensor, return_inverse, return_counts, dim);

  constexpr int64_t NoneN = 1000;
  auto dim_ = dim.has_value() ? MakeValue<int64_t>(GetValue<int64_t>(dim.value())) : MakeValue<int64_t>(NoneN);

  std::vector<AbstractBasePtr> input_abs{input_tensor->ToAbstract(), return_inverse->ToAbstract(),
                                         return_counts->ToAbstract(), dim_->ToAbstract()};

  PyBoostUtils::PrepareOpInputs(device_context, stream_id, input_tensor);
  PyBoostUtils::PrepareOpOutputs(device_context, stream_id, op->outputs());

  runtime::Pipeline::Get().WaitForward();

  const auto &outputs = op->outputs();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(device_context, input_tensor);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);

  // Get inputs kernel tensors, the not-tensor value will malloc here
  const auto &input_address_info = PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), input_abs,
                                                                input_tensor, return_inverse, return_counts, dim_);
  // Get outputs kernel tensors
  const auto &output_address_info =
    PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

  PyBoostUtils::LaunchKernel(op->primitive(), device_context, input_address_info, output_address_info);

  // update shape
  auto output_tensor_kernel = output_address_info.first;
  auto output_real_shape0 = output_tensor_kernel[kIndex0]->GetDeviceShapeVector();
  auto output_real_shape1 = output_tensor_kernel[kIndex1]->GetDeviceShapeVector();
  auto output_real_shape2 = output_tensor_kernel[kIndex2]->GetDeviceShapeVector();
  auto simple_infer_ptr = op->output_value_simple_info();
  simple_infer_ptr->shape_vector_ = ShapeArray{output_real_shape0, output_real_shape1, output_real_shape2};
  op->UpdateOutputShape(op->output(kIndex0), output_real_shape0);
  op->UpdateOutputShape(op->output(kIndex1), output_real_shape1);
  op->UpdateOutputShape(op->output(kIndex2), output_real_shape2);

  return std::make_tuple(op->output(kIndex0), op->output(kIndex1), op->output(kIndex2));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
