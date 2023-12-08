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

#include "plugin/device/cpu/kernel/pyboost/customize/copy.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "mindspore/core/ops/framework_ops.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "kernel/kernel_mod_cache.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr CopyCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "Call start";
  auto input_abs = input_tensor->ToAbstract();
  auto output_abs = input_abs->Clone();
  op->set_input_abs({input_abs});
  op->set_output_abs(output_abs);

  auto input_storage_info = input_tensor->storage_info();
  std::vector<tensor::TensorPtr> outputs;
  PyBoostUtils::CreateOutputTensor(output_abs, &outputs);
  op->set_outputs(outputs);

  PyBoostUtils::DispatchRun(std::make_shared<pynative::PyBoostDeviceTask>([op, input_tensor, input_storage_info]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();

    // Create device address for inputs
    const auto &input_addr_list = PyBoostUtils::CreateInputDeviceAddress(device_context, op->input_abs(), input_tensor);

    // Create device address for outputs
    const auto &output_addr_list = PyBoostUtils::CreateOutputDeviceAddress(device_context, op->output_abs(), outputs);

    if (input_storage_info == nullptr) {
      const auto &op_name = kTensorMoveOpName;
      const auto &inputs_kernel_tensors = PyBoostUtils::GetKernelTensorFromAddress(input_addr_list);
      const auto &outputs_kernel_tensors = PyBoostUtils::GetKernelTensorFromAddress(output_addr_list);
      // KernelMod init
      auto &cache_helper = kernel::KernelModCache::GetInstance();
      const auto &key = cache_helper.GetKernelModKey(op_name, kCPUDevice, inputs_kernel_tensors);
      auto kernel_mod = cache_helper.GetKernelMod(key);
      if (kernel_mod == nullptr) {
        kernel_mod =
          CreateKernelMod(op->primitive(), op_name, device_context, inputs_kernel_tensors, outputs_kernel_tensors);
      }
      const auto &cpu_kernel = std::dynamic_pointer_cast<kernel::NativeCpuKernelMod>(kernel_mod);
      MS_EXCEPTION_IF_NULL(cpu_kernel);
      auto thread_pool = kernel::GetActorMgrInnerThreadPool();
      cpu_kernel->SetThreadPool(thread_pool);
      // KernelMod resize
      if (cpu_kernel->Resize(inputs_kernel_tensors, outputs_kernel_tensors) == kernel::KRET_RESIZE_FAILED) {
        MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#CPU kernel op [" << op_name
                                   << "] resize failed.";
      }
      // Get workspace address
      const auto &workspace_device_address =
        PyBoostUtils::CreateWorkSpaceDeviceAddress(cpu_kernel, device_context, op_name);
      const auto &workspace_kernel_tensors = PyBoostUtils::GetKernelTensorFromAddress(workspace_device_address);
      // Do kernel launch
      if (!cpu_kernel->Launch(inputs_kernel_tensors, workspace_kernel_tensors, outputs_kernel_tensors, nullptr)) {
        MS_LOG(EXCEPTION) << "Launch kernel failed, name: " << op_name;
      }
    } else {
      if (!device_context->GetKernelExecutor(false)->ExecuteKernelTask(
            pynative::KernelTaskType::kCONTIGUOUS_TASK, input_addr_list, {input_storage_info}, output_addr_list)) {
        MS_LOG(EXCEPTION) << "ExecuteKernelTask failed, task_type:" << pynative::KernelTaskType::kCONTIGUOUS_TASK;
      }
    }

    MS_LOG(DEBUG) << "Launch end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
