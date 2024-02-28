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
#include "kernel/pyboost/py_boost_utils.h"
#include "mindspore/core/ops/framework_ops.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr CopyCustomizeCall(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, void *stream) {
  MS_LOG(DEBUG) << "Call start";
  MS_EXCEPTION_IF_NULL(input_tensor);

  auto input_abs = input_tensor->ToAbstract();
  auto output_abs = input_abs->Clone();
  op->set_input_abs({input_abs});
  op->set_output_abs(output_abs);

  std::vector<tensor::TensorPtr> outputs;
  PyBoostUtils::CreateOutputTensor(output_abs, &outputs);
  op->set_outputs(outputs);

  // Create device address for input tensors
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  // Create device address for output tensors
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, stream]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();

    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    const auto &input_device_sync = input_tensor->device_address();
    MS_EXCEPTION_IF_NULL(input_device_sync);
    if (input_device_sync->GetTensorStorageInfo() == nullptr) {
      op->set_primitive(prim::kPrimTensorMove);
      // Get inputs kernel tensors, the not-tensor value will malloc here
      const auto &input_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), input_tensor);
      // Get outputs kernel tensors
      const auto &output_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

      const auto &output_device_address =
        std::dynamic_pointer_cast<device::DeviceAddress>(op->output(0)->device_address());
      MS_EXCEPTION_IF_NULL(output_device_address);
      if (output_device_address->GetSize() != 0) {
        // Call kPrimTensorMove if input device address size if not 0.
        PyBoostUtils::LaunchKernel(op->primitive(), op->device_context(), input_address_info, output_address_info,
                                   stream);
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
  }));
  return op->output(0);
}

tensor::TensorPtr ContiguousTensorOpProcess(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  // If the tensor is continuous, return the cloned tensor and set the op information. If the tensor is not continuous,
  // return nullptr and do nothing.
  MS_EXCEPTION_IF_NULL(input_tensor);

  if (input_tensor->storage_info() == nullptr) {
    auto input_abs = input_tensor->ToAbstract();
    op->set_input_abs({input_abs});
    auto output_tensor = std::make_shared<tensor::Tensor>(*input_tensor);
    op->set_outputs({output_tensor});
    op->set_output_abs(input_abs->Clone());
    MS_LOG(DEBUG) << "Input_tensor storage_info is nullptr, just return cloned tensor, input_tensor id:"
                  << input_tensor->id() << ", output_tensor id:" << output_tensor->id();
    return output_tensor;
  }
  return nullptr;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
