/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/akg/gpu/akg_gpu_kernel_mod.h"

#include "nlohmann/json.hpp"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
using std::fstream;
using std::string;
using std::vector;

const int MAX_REGISTER_PER_THREAD_BLOCK = 65536;
const int REGISTER_UNIT_IN_WARP = 256;
const int WARP_SIZE = 32;
const int WARP_ALLOC_GRAN = 4;

GpuKernelManagerPtr GpuKernelMod::kernelmanager_ = std::make_shared<GpuKernelManager>();
GpuKernelManager::GpuKernelManager() {}

CUresult GpuKernelManager::GetFunction(const KernelPackPtr &kernel_pack, bool force_reload,
                                       vector<uint32_t> *thread_info, CUfunction *func) {
  if (kernel_pack->GetJson() == nullptr || kernel_pack->GetJson()->contents == nullptr ||
      kernel_pack->GetKernel() == nullptr || kernel_pack->GetKernel()->contents == nullptr) {
    MS_LOG(ERROR) << "GPU:Invalid kernel pack, json or kernel is nullptr.";
    return CUDA_ERROR_INVALID_IMAGE;
  }
  auto js = nlohmann::json::parse(kernel_pack->GetJson()->contents,
                                  kernel_pack->GetJson()->contents + kernel_pack->GetJson()->len);
  string fn = js["kernelName"];
  if (!force_reload) {
    auto iter = infotable_.find(fn);
    if (iter != infotable_.end()) {
      auto kernelmeta = iter->second;
      *thread_info = kernelmeta->thread_info_;
      *func = kernelmeta->func_addr_;
      return CUDA_SUCCESS;
    }
  }
  thread_info->emplace_back(js["blockIdx.x"]);
  thread_info->emplace_back(js["blockIdx.y"]);
  thread_info->emplace_back(js["blockIdx.z"]);
  thread_info->emplace_back(js["threadIdx.x"]);
  thread_info->emplace_back(js["threadIdx.y"]);
  thread_info->emplace_back(js["threadIdx.z"]);

  CUmodule module;
  CUjit_option options[1];
  options[0] = CU_JIT_MAX_REGISTERS;
  void *values[1];
  int total_threads = thread_info->at(3) * thread_info->at(4) * thread_info->at(5);
  int total_warps = std::ceil(static_cast<float>(total_threads) / static_cast<float>(WARP_SIZE));
  int limit_warps = (total_warps + WARP_ALLOC_GRAN - 1) / WARP_ALLOC_GRAN * WARP_ALLOC_GRAN;
  int total_register_unit_nums = MAX_REGISTER_PER_THREAD_BLOCK / REGISTER_UNIT_IN_WARP;
  int register_unit_nums_per_warp = total_register_unit_nums / limit_warps;
  int64_t register_nums = (register_unit_nums_per_warp * REGISTER_UNIT_IN_WARP) / WARP_SIZE;
  values[0] = reinterpret_cast<void *>(register_nums);

  CUresult result = cuModuleLoadDataEx(&module, kernel_pack->GetKernel()->contents, 1, options, values);
  if (result != CUDA_SUCCESS) {
    MS_LOG(ERROR) << "cuModuleLoadData failed.";
    return result;
  }
  result = cuModuleGetFunction(func, module, fn.c_str());
  if (result != CUDA_SUCCESS) {
    MS_LOG(ERROR) << "cuModuleGetFunction failed.";
    return result;
  }
  infotable_[fn] = std::make_shared<GpuKernelMeta>(*func, module, *thread_info);
  return result;
}

GpuKernelMod::GpuKernelMod(const KernelPackPtr &kernel_pack) : kernel_pack_(kernel_pack) {}

void GpuKernelMod::SetInputSizeList(const std::vector<size_t> &size_list) { input_size_list_ = size_list; }

void GpuKernelMod::SetOutputSizeList(const std::vector<size_t> &size_list) { output_size_list_ = size_list; }

void GpuKernelMod::SetWorkspaceSizeList(const std::vector<size_t> &size_list) { workspace_size_list_ = size_list; }

const std::vector<size_t> &GpuKernelMod::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &GpuKernelMod::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &GpuKernelMod::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool GpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == 0) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }
  if (kernel_pack_ == nullptr) {
    MS_LOG(ERROR) << "kernel pack should not be nullptr.";
    return false;
  }
  vector<uint32_t> thread_info;
  CUfunction kernel_addr;
  CUresult result = kernelmanager_->GetFunction(kernel_pack_, false, &thread_info, &kernel_addr);
  if (result != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(result, &msg);
    MS_LOG(ERROR) << "Get function failed, error: " << msg;
    return false;
  }
  std::vector<void *> runtimeargs;
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &input) -> void * { return reinterpret_cast<void *>(&(input->addr)); });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &output) -> void * { return reinterpret_cast<void *>(&(output->addr)); });
  if (!workspace.empty()) {
    (void)std::transform(std::begin(workspace), std::end(workspace), std::back_inserter(runtimeargs),
                         [](const AddressPtr &addr) -> void * { return reinterpret_cast<void *>(&(addr->addr)); });
  }
  result = cuLaunchKernel(kernel_addr, thread_info[0], thread_info[1], thread_info[2], thread_info[3], thread_info[4],
                          thread_info[5], 0, reinterpret_cast<CUstream>(stream_ptr),
                          reinterpret_cast<void **>(&runtimeargs[0]), 0);
  if (result != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(result, &msg);
    MS_LOG(ERROR) << "Launch kernel failed, error: " << msg;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
