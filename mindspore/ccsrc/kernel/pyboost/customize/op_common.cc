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
  auto input_abs = input_tensor->ToAbstract();
  auto output_abs = input_abs->Clone();
  op->set_input_abs({input_abs});
  op->set_output_abs(output_abs);

  auto input_storage_info = input_tensor->storage_info();
  std::vector<tensor::TensorPtr> outputs;
  PyBoostUtils::CreateOutputTensor(output_abs, &outputs);
  op->set_outputs(outputs);

  // Create device address for input tensors
  PyBoostUtils::PrepareOpInputs(op->device_context(), input_tensor);
  // Create device address for output tensors
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<pynative::PyBoostDeviceTask>([op, input_tensor, input_storage_info, stream]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();

      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      if (input_storage_info == nullptr) {
        op->set_primitive(prim::kPrimTensorMove);
        // Get inputs kernel tensors, the not-tensor value will malloc here
        const auto &input_address_info = PyBoostUtils::GetAddressInfo(device_context, op->input_abs(), input_tensor);
        // Get outputs kernel tensors
        const auto &output_address_info = PyBoostUtils::GetAddressInfo(device_context, {op->output_abs()}, outputs);

        PyBoostUtils::LaunchKernel(op->primitive(), op->device_context(), input_address_info, output_address_info,
                                   stream);
      } else {
        const auto &input_address = std::dynamic_pointer_cast<device::DeviceAddress>(input_tensor->device_address());
        const auto &output_address = std::dynamic_pointer_cast<device::DeviceAddress>(op->output(0)->device_address());
        if (!device_context->GetKernelExecutor(false)->ExecuteKernelTask(pynative::KernelTaskType::kCONTIGUOUS_TASK,
                                                                         {input_address}, {input_storage_info},
                                                                         {output_address}, op->stream_id())) {
          MS_LOG(EXCEPTION) << "ExecuteKernelTask failed, task_type:" << pynative::KernelTaskType::kCONTIGUOUS_TASK;
        }
      }

      MS_LOG(DEBUG) << "Launch end";
    }));
  return op->output(0);
}  // namespace pyboost
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
