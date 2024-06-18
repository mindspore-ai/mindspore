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

#include "kernel/pyboost/customize/op_common.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "mindspore/core/ops/framework_ops.h"
#include "kernel/pyboost/auto_generate/maximum.h"
#include "kernel/pyboost/auto_generate/minimum.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::BaseTensorPtr CopyCustomizeCall(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "Call start";
  MS_EXCEPTION_IF_NULL(input_tensor);

  auto input_abs = input_tensor->ToAbstract();
  input_abs->set_value(kValueAny);
  auto output_abs = input_abs->Clone();
  op->set_input_abs({input_abs});
  op->set_output_abs(output_abs);

  std::vector<tensor::BaseTensorPtr> outputs;
  PyBoostUtils::CreateOutputTensor(output_abs, &outputs);
  op->set_outputs(outputs);

  // Create device address for input tensors
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  // Create device address for output tensors
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  runtime::Pipeline::Get().WaitForward();
  auto device_context = op->device_context();
  const auto &op_outputs = op->outputs();

  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(device_context, input_tensor);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, op_outputs);

  const auto &input_device_sync = input_tensor->device_address();
  MS_EXCEPTION_IF_NULL(input_device_sync);
  if (input_device_sync->GetTensorStorageInfo() == nullptr) {
    op->set_primitive(prim::kPrimTensorMove);
    // Get inputs kernel tensors, the not-tensor value will malloc here
    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), input_tensor);
    // Get outputs kernel tensors
    const auto &output_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, op_outputs);

    const auto &output_device_address =
      std::dynamic_pointer_cast<device::DeviceAddress>(op->output(0)->device_address());
    MS_EXCEPTION_IF_NULL(output_device_address);
    if (output_device_address->GetSize() != 0) {
      // Call kPrimTensorMove if input device address size if not 0.
      PyBoostUtils::LaunchKernel(op->primitive(), op->device_context(), input_address_info, output_address_info,
                                 op->stream_id());
    }
  } else {
    const auto &input_address = std::dynamic_pointer_cast<device::DeviceAddress>(input_tensor->device_address());
    const auto &output_address = std::dynamic_pointer_cast<device::DeviceAddress>(op->output(0)->device_address());
    if (!device_context->GetKernelExecutor(false)->ExecuteKernelTask(
          runtime::KernelTaskType::kCONTIGUOUS_TASK, {input_address}, {output_address}, op->stream_id())) {
      MS_LOG(EXCEPTION) << "ExecuteKernelTask failed, task_type:" << runtime::KernelTaskType::kCONTIGUOUS_TASK;
    }
  }

  MS_LOG(DEBUG) << "Launch end";
  return op->output(0);
}

tensor::BaseTensorPtr ContiguousTensorOpProcess(const std::shared_ptr<OpRunner> &op,
                                                const BaseTensorPtr &input_tensor) {
  // If the tensor is continuous, return the cloned tensor and set the op information. If the tensor is not continuous,
  // return nullptr and do nothing.
  MS_EXCEPTION_IF_NULL(input_tensor);

  if (input_tensor->storage_info() == nullptr) {
    auto input_abs = input_tensor->ToAbstract();
    input_abs->set_value(kValueAny);
    op->set_input_abs({input_abs});
    auto output_tensor = std::make_shared<tensor::BaseTensor>(*input_tensor);
    op->set_outputs({output_tensor});
    op->set_output_abs(input_abs->Clone());
    MS_LOG(DEBUG) << "Input_tensor storage_info is nullptr, just return cloned tensor, input_tensor id:"
                  << input_tensor->id() << ", output_tensor id:" << output_tensor->id();
    return output_tensor;
  }
  return nullptr;
}

tensor::BaseTensorPtr ClampTensorCustomizeCall(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor,
                                               const std::optional<BaseTensorPtr> &min,
                                               const std::optional<BaseTensorPtr> &max,
                                               const std::string &device_target) {
  MS_LOG(DEBUG) << "Call ClampTensor start";
  if (!min.has_value() && !max.has_value()) {
    MS_EXCEPTION(ValueError) << "For Clamp, at least one of 'min' or 'max' must not be None.";
  }
  auto device_context = op->device_context();

  std::vector<AbstractBasePtr> input_abs = {x_tensor->ToAbstract()};
  OpPtr final_node = nullptr;

  BaseTensorPtr output = x_tensor;
  if (min.has_value()) {
    input_abs.emplace_back(min.value()->ToAbstract());
    auto min_tensor = PyBoostUtils::CastTensor(min.value(), x_tensor->Dtype()->type_id(), device_target);
    const auto &maximum = CREATE_PYBOOST_OP(Maximum, device_context->device_context_key_.device_name_);
    output = maximum->Call(output, min_tensor);
    final_node = maximum;
  } else {
    input_abs.emplace_back(std::make_shared<abstract::AbstractNone>());
  }
  if (max.has_value()) {
    input_abs.emplace_back(max.value()->ToAbstract());
    auto max_tensor = PyBoostUtils::CastTensor(max.value(), x_tensor->Dtype()->type_id(), device_target);
    const auto &minimum = CREATE_PYBOOST_OP(Minimum, device_context->device_context_key_.device_name_);
    output = minimum->Call(output, max_tensor);
    final_node = minimum;
  } else {
    input_abs.emplace_back(std::make_shared<abstract::AbstractNone>());
  }

  op->set_input_abs(input_abs);
  op->set_output_abs(final_node->output_abs());
  op->set_outputs({output});
  MS_LOG(DEBUG) << "Call ClampTensor end";
  return output;
}

tensor::BaseTensorPtr ClampScalarCustomizeCall(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor,
                                               const std::optional<ScalarPtr> &min, const std::optional<ScalarPtr> &max,
                                               const std::string &device_target) {
  MS_LOG(DEBUG) << "Call ClampScalar start";
  if (!min.has_value() && !max.has_value()) {
    MS_EXCEPTION(ValueError) << "For Clamp, at least one of 'min' or 'max' must not be None.";
  }
  auto device_context = op->device_context();

  std::vector<AbstractBasePtr> input_abs = {x_tensor->ToAbstract()};
  OpPtr final_node = nullptr;

  BaseTensorPtr output = x_tensor;
  if (min.has_value()) {
    input_abs.emplace_back(min.value()->ToAbstract());
    auto min_tensor = PyBoostUtils::ScalarToTensor(min.value());
    min_tensor = PyBoostUtils::CastTensor(min_tensor, x_tensor->Dtype()->type_id(), device_target);
    const auto &maximum = CREATE_PYBOOST_OP(Maximum, device_context->device_context_key_.device_name_);
    output = maximum->Call(output, min_tensor);
    final_node = maximum;
  } else {
    input_abs.emplace_back(std::make_shared<abstract::AbstractNone>());
  }
  if (max.has_value()) {
    input_abs.emplace_back(max.value()->ToAbstract());
    auto max_tensor = PyBoostUtils::ScalarToTensor(max.value());
    max_tensor = PyBoostUtils::CastTensor(max_tensor, x_tensor->Dtype()->type_id(), device_target);
    const auto &minimum = CREATE_PYBOOST_OP(Minimum, device_context->device_context_key_.device_name_);
    output = minimum->Call(output, max_tensor);
    final_node = minimum;
  } else {
    input_abs.emplace_back(std::make_shared<abstract::AbstractNone>());
  }

  op->set_input_abs(input_abs);
  op->set_output_abs(final_node->output_abs());
  op->set_outputs({output});

  MS_LOG(DEBUG) << "Call ClampScalar end";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
