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

#include "backend/kernel_compiler/tbe/tbe_kernel_mod.h"
#include <algorithm>
#include "runtime/rt.h"
#include "utils/ms_context.h"
#include "graphengine/inc/framework/ge_runtime/task_info.h"
#include "runtime/device/ascend/executor/ai_core_dynamic_kernel.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
using TbeTaskInfoPtr = std::shared_ptr<ge::model_runner::TbeTaskInfo>;
using tbe::KernelManager;
using AddressPtrList = std::vector<mindspore::kernel::AddressPtr>;
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
  const void *stubFunc = reinterpret_cast<void *>(func_stub);
  auto argsSize = static_cast<uint32_t>(UlongToUint(sizeof(void *)) * runtimeargs.size());
  if (RT_ERROR_NONE != rtKernelLaunch(stubFunc, blockdim, runtimeargs.data(), argsSize, l2ctrl, stream_ptr)) {
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
                       [](const AddressPtr &input) -> void * { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(output_data_addrs),
                       [](const AddressPtr &output) -> void * { return output->addr; });
  if (!workspaces.empty()) {
    (void)std::transform(std::begin(workspaces), std::end(workspaces), std::back_inserter(workspace_addrs),
                         [](const AddressPtr &workspace) -> void * { return workspace->addr; });
  }

  stream_id_ = stream_id;
  auto funcstub = KernelManager::GenFuncStub(*kernel_pack_, false, &block_dim_);
  if (funcstub == 0) {
    MS_EXCEPTION(ArgumentError) << "GenFuncStub failed.";
  }

  std::string stub_func = KernelManager::GetStubFuncName(kernel_pack_);

  MS_LOG(INFO) << "block_dim is:" << block_dim_;

  TbeTaskInfoPtr task_info_ptr = make_shared<ge::model_runner::TbeTaskInfo>(
    kernel_name_, stream_id, stub_func, block_dim_, args, 0, sm_desc, nullptr, 0, meta_data, input_data_addrs,
    output_data_addrs, workspace_addrs, NeedDump());
  return {task_info_ptr};
}

device::DynamicKernelPtr TbeKernelMod::GenDynamicKernel(const CNodePtr &cnode_ptr, void *stream_ptr) {
  AddressPtrList kernel_inputs;
  AddressPtrList kernel_workspaces;
  AddressPtrList kernel_outputs;
  device::KernelRuntime::GenLaunchArgs(*this, cnode_ptr, &kernel_inputs, &kernel_workspaces, &kernel_outputs);
  auto dynamic_flag = AnfAlgo::IsDynamicShape(cnode_ptr);

  // Get para_size from json
  auto kernel_json_info = kernel_pack_->kernel_json_info();
  auto op_para_size = kernel_json_info.op_para_size;

  // Generate args
  std::vector<void *> runtime_args;
  (void)std::transform(std::begin(kernel_inputs), std::end(kernel_inputs), std::back_inserter(runtime_args),
                       [](const AddressPtr &input) -> void * { return input->addr; });
  (void)std::transform(std::begin(kernel_outputs), std::end(kernel_outputs), std::back_inserter(runtime_args),
                       [](const AddressPtr &output) -> void * { return output->addr; });
  if (!kernel_workspaces.empty()) {
    (void)std::transform(std::begin(kernel_workspaces), std::end(kernel_workspaces), std::back_inserter(runtime_args),
                         [](const AddressPtr &addr) -> void * { return addr->addr; });
  }

  void *tiling_data_ptr = nullptr;
  if (op_para_size > 0) {
    auto ret = rtMalloc(&tiling_data_ptr, op_para_size, RT_MEMORY_HBM);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "rtMalloc tiling data failed";
    }
    runtime_args.push_back(tiling_data_ptr);
  }

  // Get stub_function
  uint32_t block_dim = 1;  // default blockdim equal to 1.
  device::DynamicKernelPtr executor = nullptr;
  std::string origin_key;
  void *handle = nullptr;
  auto func_stub = KernelManager::GenFuncStub(*kernel_pack_, false, &block_dim, dynamic_flag, &handle, &origin_key);
  if (dynamic_flag) {
    if (func_stub != 1) {
      MS_LOG(EXCEPTION) << "GenFuncStub failed.";
    }
    executor = std::make_shared<device::ascend::AiCoreDynamicKernel>(handle, block_dim, tiling_data_ptr, op_para_size,
                                                                     stream_ptr, cnode_ptr, runtime_args, origin_key);
  } else {
    if (func_stub == 0) {
      MS_LOG(EXCEPTION) << "GenFuncStub failed.";
    }
    const void *stub_func_ptr = reinterpret_cast<void *>(func_stub);
    executor = std::make_shared<device::ascend::AiCoreDynamicKernel>(stub_func_ptr, block_dim, tiling_data_ptr,
                                                                     op_para_size, stream_ptr, cnode_ptr, runtime_args);
  }
  return executor;
}

vector<size_t> TbeKernelMod::GenParameters() {
  auto kernel_json_info = kernel_pack_->kernel_json_info();
  return kernel_json_info.parameters;
}
}  // namespace kernel
}  // namespace mindspore
