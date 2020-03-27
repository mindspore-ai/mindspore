/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "kernel/tbe/tbe_kernel_mod.h"
#include <algorithm>
#include "runtime/rt.h"
#include "nlohmann/json.hpp"
#include "graphengine/inc/framework/ge_runtime/task_info.h"

namespace mindspore {
namespace kernel {
using TbeTaskInfoPtr = std::shared_ptr<ge::model_runner::TbeTaskInfo>;
using tbe::KernelManager;
bool TbeKernelMod::Launch(const std::vector<mindspore::kernel::AddressPtr> &inputs,
                          const std::vector<mindspore::kernel::AddressPtr> &workspace,
                          const std::vector<mindspore::kernel::AddressPtr> &outputs, uintptr_t stream_ptr) {
  if (stream_ptr == 0) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }

  if (kernel_pack_ == nullptr) {
    MS_LOG(ERROR) << "kernel pack should not be nullptr.";
    return false;
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
                       [](const AddressPtr &input) -> void * { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &output) -> void * { return output->addr; });
  if (!workspace.empty()) {
    (void)std::transform(std::begin(workspace), std::end(workspace), std::back_inserter(runtimeargs),
                         [](const AddressPtr &addr) -> void * { return addr->addr; });
  }
  rtL2Ctrl_t *l2ctrl = nullptr;
  auto *stream = reinterpret_cast<rtStream_t *>(stream_ptr);
  const void *stubFunc = reinterpret_cast<void *>(func_stub);
  auto argsSize = static_cast<uint32_t>(UlongToUint(sizeof(void *)) * runtimeargs.size());
  if (RT_ERROR_NONE != rtKernelLaunch(stubFunc, blockdim, runtimeargs.data(), argsSize, l2ctrl, stream)) {
    MS_LOG(ERROR) << "Call runtime rtKernelLaunch error.";
    return false;
  }

  return true;
}

vector<TaskInfoPtr> TbeKernelMod::GenTask(const std::vector<AddressPtr> &inputs,
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
                       [](const AddressPtr &input) -> void * { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(output_data_addrs),
                       [](const AddressPtr &output) -> void * { return output->addr; });
  if (!workspaces.empty()) {
    (void)std::transform(std::begin(workspaces), std::end(workspaces), std::back_inserter(workspace_addrs),
                         [](const AddressPtr &workspace) -> void * { return workspace->addr; });
  }

  uint32_t block_dim = 1;  // default blockdim equal to 1.
  auto funcstub = KernelManager::GenFuncStub(*kernel_pack_, false, &block_dim);
  if (funcstub == 0) {
    MS_EXCEPTION(ArgumentError) << "GenFuncStub failed.";
  }

  std::string stub_func = KernelManager::GetStubFuncName(kernel_pack_);

  MS_LOG(INFO) << "block_dim is:" << block_dim;

  TbeTaskInfoPtr task_info_ptr =
    make_shared<ge::model_runner::TbeTaskInfo>(stream_id, stub_func, block_dim, args, 0, sm_desc, nullptr, 0, meta_data,
                                               input_data_addrs, output_data_addrs, workspace_addrs);
  return {task_info_ptr};
}

vector<size_t> TbeKernelMod::GenParameters() {
  auto kernel_json_info = kernel_pack_->kernel_json_info();
  return kernel_json_info.parameters;
}
}  // namespace kernel
}  // namespace mindspore
