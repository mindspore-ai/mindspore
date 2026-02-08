/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/runtime/actor/debug_actor.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#ifdef ENABLE_DEBUGGER
#include "backend/ge_backend/dump/hook_debugger.h"
#endif
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "tools/profiler/profiling.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {

void DebugActor::DebugPreLaunch(const AnfNodePtr &node, const std::vector<KernelTensorPtr> &input_kernel_tensors,
                                const std::vector<KernelTensorPtr> &output_kernel_tensors,
                                OpContext<KernelTensor> *const op_context, const AID *) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(op_context);
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU.
 * Runtime category: MindRT.
 * Description: Load and read data for the given node if needed. Dump the node if dump is enabled and free the loaded
 * memory after the dump (for GPU and ascend kernel-by-kernel).
 */
void DebugActor::DebugPostLaunch(const AnfNodePtr &node, const std::vector<KernelTensorPtr> &input_kernel_tensors,
                                 const std::vector<KernelTensorPtr> &output_kernel_tensors,
                                 OpContext<KernelTensor> *const op_context, const AID *) {
  std::vector<KernelTensor *> raw_input_kernel_tensors;
  raw_input_kernel_tensors.resize(input_kernel_tensors.size());
  std::vector<KernelTensor *> raw_output_kernel_tensors;
  raw_output_kernel_tensors.resize(output_kernel_tensors.size());
  std::transform(input_kernel_tensors.begin(), input_kernel_tensors.end(), raw_input_kernel_tensors.begin(),
                 [](const KernelTensorPtr &ptr) { return ptr.get(); });
  std::transform(output_kernel_tensors.begin(), output_kernel_tensors.end(), raw_output_kernel_tensors.begin(),
                 [](const KernelTensorPtr &ptr) { return ptr.get(); });
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(op_context);
}
void DebugActor::DebugOnStepBegin(const std::vector<KernelGraphPtr> &graphs,
                                  const std::vector<AnfNodePtr> &origin_parameters_order,
                                  OpContext<KernelTensor> *const op_context, const AID *) {
  MS_LOG(INFO) << "Debug on step begin.";
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if (profiler == nullptr || !profiler->IsInitialized()) {
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto &hookDebugger = dump::HookDebugger::GetInstance();
    if (hookDebugger.IsHookerEnabled()) {
      MS_LOG(INFO) << "On multi graph step begin, hookdebugger is enable.";
      hookDebugger.HookOnStepBegin(device_id, graphs, step_count_, false);
    }
  }
}

void DebugActor::DebugOnStepEnd(OpContext<KernelTensor> *const, const AID *, int total_running_count, int sink_size) {
  MS_LOG(INFO) << "Debug on step end. total_running_count is: " << total_running_count;
  step_count_ = total_running_count;
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  auto &hookDebugger = dump::HookDebugger::GetInstance();
  if (hookDebugger.IsHookerEnabled()) {
    MS_LOG(INFO) << "On step end, hookdebugger is enable.";
    host_context->device_res_manager_->SyncAllStreams(false);
    hookDebugger.HookOnStepEnd();
  }
  host_context->device_res_manager_->SyncAllStreams(false);
}
void DebugActor::Finalize() {}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
