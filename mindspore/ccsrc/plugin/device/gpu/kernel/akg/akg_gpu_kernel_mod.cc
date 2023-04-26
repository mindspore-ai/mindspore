/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/akg/akg_gpu_kernel_mod.h"

#include <algorithm>
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
const int AKG_KERNEL_MOD_BX_IDX = 0;
const int AKG_KERNEL_MOD_BY_IDX = 1;
const int AKG_KERNEL_MOD_BZ_IDX = 2;
const int AKG_KERNEL_MOD_TX_IDX = 3;
const int AKG_KERNEL_MOD_TY_IDX = 4;
const int AKG_KERNEL_MOD_TZ_IDX = 5;

AkgGpuKernelManagerPtr AkgGpuKernelMod::kernel_manager_ = std::make_shared<AkgGpuKernelManager>();
AkgGpuKernelManager::AkgGpuKernelManager() {}

CUresult AkgGpuKernelManager::GetFunction(const KernelPackPtr &kernel_pack, bool force_reload,
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
    const char *msg = nullptr;
    cuGetErrorName(result, &msg);
    MS_LOG(ERROR) << "cuModuleLoadDataEx failed. Kernel name: << " << fn << ". Error message: " << msg;
    return result;
  }
  result = cuModuleGetFunction(func, module, fn.c_str());
  if (result != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(result, &msg);
    MS_LOG(ERROR) << "cuModuleGetFunction failed. Kernel name: << " << fn << ". Error message: " << msg;
    return result;
  }
  infotable_[fn] = std::make_shared<GpuKernelMeta>(*func, module, *thread_info);
  return result;
}

AkgGpuKernelMod::AkgGpuKernelMod(const KernelPackPtr &kernel_pack) : kernel_pack_(kernel_pack) {
  if (kernel_pack != nullptr) {
    auto js = kernel_pack->GetJson();
    if (js != nullptr) {
      auto parsed_js = nlohmann::json::parse(js->contents, js->contents + js->len);
      kernel_name_ = parsed_js["kernelName"];
    }
  }
}

bool AkgGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                             const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == 0) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr. Kernel name: " << kernel_name_;
    return false;
  }
  if (kernel_pack_ == nullptr) {
    MS_LOG(ERROR) << "kernel pack should not be nullptr. Kernel name: " << kernel_name_;
    return false;
  }
  CUresult result;
  if (kernel_addr_ == nullptr) {
    result = kernel_manager_->GetFunction(kernel_pack_, false, &thread_info_, &kernel_addr_);
    if (result != CUDA_SUCCESS) {
      const char *msg = nullptr;
      cuGetErrorName(result, &msg);
      MS_LOG(ERROR) << "Get function " << kernel_name_ << " failed. Error message: " << msg;
      return false;
    }
  }
  std::vector<void *> runtimeargs;
  runtimeargs.reserve(inputs.size() + outputs.size() + workspace.size());
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &input) { return reinterpret_cast<void *>(&(input->addr)); });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &output) { return reinterpret_cast<void *>(&(output->addr)); });
  if (!workspace.empty()) {
    (void)std::transform(std::begin(workspace), std::end(workspace), std::back_inserter(runtimeargs),
                         [](const AddressPtr &addr) { return reinterpret_cast<void *>(&(addr->addr)); });
  }
  result = cuLaunchKernel(kernel_addr_, thread_info_[AKG_KERNEL_MOD_BX_IDX], thread_info_[AKG_KERNEL_MOD_BY_IDX],
                          thread_info_[AKG_KERNEL_MOD_BZ_IDX], thread_info_[AKG_KERNEL_MOD_TX_IDX],
                          thread_info_[AKG_KERNEL_MOD_TY_IDX], thread_info_[AKG_KERNEL_MOD_TZ_IDX], 0,
                          reinterpret_cast<CUstream>(stream_ptr), reinterpret_cast<void **>(&runtimeargs[0]), 0);
  if (result != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(result, &msg);
    MS_LOG(ERROR) << "Launch kernel failed. Kernel name: " << kernel_name_ << ". cuLaunchKernel error message: " << msg;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
