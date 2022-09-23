/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/tbe/tbe_kernel_mod.h"

#include <algorithm>
#include "runtime/rt.h"
#include "utils/ms_context.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task_info.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
using TbeTaskInfoPtr = std::shared_ptr<mindspore::ge::model_runner::TbeTaskInfo>;
using tbe::KernelManager;
bool TbeKernelMod::Launch(const std::vector<mindspore::kernel::AddressPtr> &inputs,
                          const std::vector<mindspore::kernel::AddressPtr> &workspace,
                          const std::vector<mindspore::kernel::AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }

  if (kernel_pack_ == nullptr) {
    MS_LOG(ERROR) << "kernel pack should not be nullptr.";
    return false;
  }
  // launch atomic_cleans first
  if (!atomic_clean_nodes_.empty()) {
    for (const auto &atomic_clean_node : atomic_clean_nodes_) {
      KernelLaunchInfo kernel_launch_info;
      auto kernel_mod = AnfAlgo::GetKernelMod(atomic_clean_node.lock());
      MS_EXCEPTION_IF_NULL(kernel_mod);
      device::KernelRuntime::GenLaunchArgs(*kernel_mod, atomic_clean_node.lock(), &kernel_launch_info);
      auto atomic_inputs = kernel_launch_info.inputs_;
      std::vector<AddressPtr> atomic_outputs;
      std::vector<AddressPtr> atomic_workspace;
      kernel_mod->Launch(atomic_inputs, atomic_workspace, atomic_outputs, stream_ptr);
    }
  }

  uint32_t blockdim = 1;  // default blockdim equal to 1.
  auto func_stub = KernelManager::GenFuncStub(*kernel_pack_, false, &blockdim);
  if (func_stub == 0) {
    MS_LOG(ERROR) << "GenFuncStub failed.";
    return false;
  }

  // pack all addresses into a vector.
  std::vector<void *> runtimeargs;
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &input) -> void * {
                         MS_EXCEPTION_IF_NULL(input);
                         return input->addr;
                       });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &output) -> void * {
                         MS_EXCEPTION_IF_NULL(output);
                         return output->addr;
                       });
  if (!workspace.empty()) {
    (void)std::transform(std::begin(workspace), std::end(workspace), std::back_inserter(runtimeargs),
                         [](const AddressPtr &addr) -> void * {
                           MS_EXCEPTION_IF_NULL(addr);
                           return addr->addr;
                         });
  }
  rtL2Ctrl_t *l2ctrl = nullptr;
  const void *stubFunc = reinterpret_cast<void *>(func_stub);
  auto argsSize = static_cast<uint32_t>(UlongToUint(sizeof(void *)) * runtimeargs.size());
  auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
  auto ret = rtKernelLaunch(stubFunc, blockdim, runtimeargs.data(), argsSize, l2ctrl, stream_ptr);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call runtime rtKernelLaunch error.";
    return false;
  }

  return true;
}

std::vector<TaskInfoPtr> TbeKernelMod::GenTask(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &workspaces,
                                               const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  if (kernel_pack_ == nullptr) {
    MS_EXCEPTION(ArgumentError) << "kernel pack should not be nullptr.";
  }

  std::vector<uint8_t> args;
  std::vector<uint8_t> sm_desc;
  std::vector<uint8_t> meta_data;
  std::vector<void *> input_data_addrs;
  std::vector<void *> output_data_addrs;
  std::vector<void *> workspace_addrs;

  // pack all addresses into a vector.
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(input_data_addrs),
                       [](const AddressPtr &input) -> void * {
                         MS_EXCEPTION_IF_NULL(input);
                         return input->addr;
                       });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(output_data_addrs),
                       [](const AddressPtr &output) -> void * {
                         MS_EXCEPTION_IF_NULL(output);
                         return output->addr;
                       });
  if (!workspaces.empty()) {
    (void)std::transform(std::begin(workspaces), std::end(workspaces), std::back_inserter(workspace_addrs),
                         [](const AddressPtr &workspace) -> void * {
                           MS_EXCEPTION_IF_NULL(workspace);
                           return workspace->addr;
                         });
  }

  stream_id_ = stream_id;
  auto funcstub = KernelManager::GenFuncStub(*kernel_pack_, false, &block_dim_);
  if (funcstub == 0) {
    MS_EXCEPTION(ArgumentError) << "GenFuncStub failed.";
  }

  std::string stub_func = KernelManager::GetStubFuncName(kernel_pack_);

  MS_LOG(DEBUG) << "block_dim is:" << block_dim_;

  TbeTaskInfoPtr task_info_ptr = std::make_shared<mindspore::ge::model_runner::TbeTaskInfo>(
    unique_name_, stream_id, stub_func, block_dim_, args, 0, sm_desc, nullptr, 0, meta_data, input_data_addrs,
    output_data_addrs, workspace_addrs, NeedDump());
  return {task_info_ptr};
}

vector<size_t> TbeKernelMod::GenParameters() {
  MS_EXCEPTION_IF_NULL(kernel_pack_);
  auto kernel_json_info = kernel_pack_->kernel_json_info();
  return kernel_json_info.parameters;
}
}  // namespace kernel
}  // namespace mindspore
