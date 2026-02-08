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

#include "kernel/gpu/dynamic_akg/dynamic_akg_gpu_kernel_mod.h"
#include <fstream>
#include <numeric>
#include <functional>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <map>
#include "nlohmann/json.hpp"
#include "utils/ms_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "include/utils/common.h"
#include "kernel/gpu/gpu_common.h"

namespace mindspore {
namespace kernel {
using std::fstream;
using std::string;
using std::unordered_map;
using std::vector;

constexpr auto kStaticTileImpl = "StaticTileImpl";
constexpr auto kSupportInfo = "SupportInfo";

constexpr int64_t kRemove = -100000;
constexpr int64_t kKeep = -99999;

DynamicAkgGpuKernelManagerPtr DynamicAkgGpuKernelMod::kernel_manager_ = std::make_shared<DynamicAkgGpuKernelManager>();
DynamicAkgGpuKernelManager::DynamicAkgGpuKernelManager() {}

CUresult DynamicAkgGpuKernelManager::GetCUResult(const char *kernel_content, bool force_reload,
                                                 vector<uint32_t> *thread_info, CUfunction *func,
                                                 const string kernel_name) {
  string fn = kernel_name;
  CUmodule module;
  CUjit_option options[] = {};
  void *optionValues[] = {};
  CUresult result = cuModuleLoadDataEx(&module, kernel_content, 0, options, optionValues);
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

CUresult DynamicAkgGpuKernelManager::GetFunction(const KernelPackPtr &kernel_pack, bool force_reload,
                                                 vector<uint32_t> *thread_info, CUfunction *func,
                                                 const string kernel_name) {
  if (kernel_pack->GetJson() == nullptr || kernel_pack->GetJson()->contents == nullptr ||
      kernel_pack->GetKernel() == nullptr || kernel_pack->GetKernel()->contents == nullptr) {
    MS_LOG(ERROR) << "Invalid kernel pack, json or kernel is nullptr of kernel : " << kernel_name << ".\n";
    return CUDA_ERROR_INVALID_IMAGE;
  }
  return GetCUResult(&kernel_pack->GetKernel()->contents[0], force_reload, thread_info, func, kernel_name);
}

void DynamicAkgGpuKernelMod::UpdateShapeList(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  shape_list_.clear();
  for (size_t i = 0; i < inputs.size(); i++) {
    auto in_shape = inputs[i]->GetShapeVector();
    (void)shape_list_.emplace_back(in_shape);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    auto out_shape = outputs[i]->GetShapeVector();
    (void)shape_list_.emplace_back(out_shape);
  }
  for (auto it : kernel_map_) {
    it.second->shape_list_ = shape_list_;
  }
  MS_LOG(INFO) << "Done UpdateShapeList for " << kernel_name_ << " shape_list_ = " << shape_list_;
}

void DynamicAkgGpuKernelMod::UpdateStaticShapeMappingInfo() {
  thread_info_.clear();
  thread_info_.emplace_back(parsed_js_[kBlockIdxX]);
  thread_info_.emplace_back(parsed_js_[kBlockIdxY]);
  thread_info_.emplace_back(parsed_js_[kBlockIdxZ]);
  thread_info_.emplace_back(parsed_js_[kThreadIdxX]);
  thread_info_.emplace_back(parsed_js_[kThreadIdxY]);
  thread_info_.emplace_back(parsed_js_[kThreadIdxZ]);
  MS_LOG(INFO) << "Done UpdateStaticShapeMappingInfo for " << kernel_name_;
}

DynamicAkgGpuKernelMod::DynamicAkgGpuKernelMod(const KernelPackPtr &kernel_pack) : kernel_pack_(kernel_pack) {
  if (kernel_pack != nullptr) {
    auto js = kernel_pack->GetJson();
    if (js != nullptr) {
      parsed_js_ = nlohmann::json::parse(js->contents, js->contents + js->len);
      kernel_name_ = parsed_js_["kernelName"];
    }
  }
}

void DynamicAkgGpuKernelMod::CheckJsonParsed() {
  if (parsed_js_ != nullptr) {
    return;
  }
  if (kernel_pack_ == nullptr) {
    MS_EXCEPTION(RuntimeError) << "Invalid kernel pack for kernel: " << kernel_name_ << ".";
  }
  auto js = kernel_pack_->GetJson();
  if (js == nullptr) {
    MS_EXCEPTION(RuntimeError) << "Invalid kernel pack, json is nullptr for kernel: " << kernel_name_ << ".";
  }
  parsed_js_ = nlohmann::json::parse(js->contents, js->contents + js->len);
}

void DynamicAkgGpuKernelMod::Initialize() {
  CheckJsonParsed();
  if (is_dynamic_) {
    InitAkgKernelImpls();
  } else {
    UpdateStaticShapeMappingInfo();
    const size_t thread_info_num = 6;
    if (thread_info_.size() != thread_info_num ||
        (std::any_of(thread_info_.begin(), thread_info_.end(), [](uint32_t t) { return t <= 0; }))) {
      MS_EXCEPTION(ValueError) << "For " << kernel_name_
                               << ", gpu mapping config must be updated to 6 positive numbers before "
                               << "launch, but got thread_info = " << thread_info_;
    }
  }
}

void DynamicAkgGpuKernelMod::InitAkgKernelImpls() {
  kernel_map_[kernel_name_] = std::make_shared<DynamicTileImpl>(kernel_name_, parsed_js_);
  if (parsed_js_.find(kStaticTileImpl) != parsed_js_.end()) {
    auto static_kernel_json = parsed_js_.at(kStaticTileImpl);
    auto static_kernel_name = static_kernel_json["kernelName"];
    kernel_map_[static_kernel_name] = std::make_shared<StaticTileImpl>(static_kernel_name, static_kernel_json);
  }
  for (auto it : kernel_map_) {
    it.second->InitJsonShapeInformation();
    it.second->InitJsonMappingInformation();
    if (it.second->parsed_js_[kSupportInfo]["OperatorType"] == "Reduce") {
      it.second->preprocessDynamicReduceTiling();
    }
  }
  MS_LOG(INFO) << "InitAkgKernelImpls " << kernel_map_.size();
}

AkgKernelImplInfoPtr DynamicAkgGpuKernelMod::SelectKernelImpl() {
  if (kernel_map_.find(kernel_name_) == kernel_map_.end()) {
    MS_EXCEPTION(RuntimeError) << "No default kernel for " << kernel_name_;
  }
  auto default_kernel = kernel_map_[kernel_name_];
  AkgKernelImplInfoPtr static_kernel = nullptr;
  for (auto it : kernel_map_) {
    if (it.second == default_kernel) {
      continue;
    }
    static_kernel = it.second;
  }
  if (static_kernel == nullptr) {
    MS_LOG(DEBUG) << "For " << kernel_name_ << ", only have default kernel, return";
    return default_kernel;
  }
  if (default_kernel->runtime_vars_.empty()) {
    MS_LOG(DEBUG) << "For " << kernel_name_ << ", default kernel is static tile, return";
    return default_kernel;
  }
  static_kernel->Init();
  default_kernel->Init();
  for (auto it : default_kernel->runtime_vars_) {
    MS_LOG(INFO) << "Runtime var: " << it.second->ToString();
    bool is_thread = it.second->curr_map_id >= AKG_KERNEL_MOD_TX_IDX && it.second->curr_map_id <= AKG_KERNEL_MOD_TZ_IDX;
    const int64_t upper_bound_limit = 32;
    if (is_thread && it.second->upper_bound <= upper_bound_limit) {
      return static_kernel;
    }
  }
  MS_LOG(INFO) << kernel_name_ << " use dynamic tile kernel, shape " << shape_list_ << "; Static thread info "
               << static_kernel->thread_info_;
  return kernel_map_[kernel_name_];
}

int DynamicAkgGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  UpdateShapeList(inputs, outputs);
  kernel_impl_ = SelectKernelImpl();
  kernel_impl_->Resize();
  MS_LOG(INFO) << "Done resize for DynamicAkgGpuKernelMod for " << kernel_name_;
  return ret;
}

bool DynamicAkgGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (kernel_impl_ != nullptr) {
    kernel_name_ = kernel_impl_->kernel_name_;
    thread_info_ = kernel_impl_->thread_info_;
    arg_size_vec_ = kernel_impl_->arg_size_vec_;
  }

  if (stream_ptr == 0) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr. Kernel name: " << kernel_name_;
    return false;
  }
  if (kernel_pack_ == nullptr) {
    MS_LOG(ERROR) << "kernel pack should not be nullptr. Kernel name: " << kernel_name_;
    return false;
  }
  MS_LOG(INFO) << "Start Launch for " << kernel_name_;
  CUresult result;
  if (kernel_addr_ == nullptr) {
    result = kernel_manager_->GetFunction(kernel_pack_, false, &thread_info_, &kernel_addr_, kernel_name_);
    if (result != CUDA_SUCCESS) {
      const char *msg = nullptr;
      cuGetErrorName(result, &msg);
      MS_LOG(ERROR) << "Get function " << kernel_name_ << " failed. Error message: " << msg;
      return false;
    }
  }

  vector<void *> runtimeargs;
  vector<void *> dev_addrs;
  vector<size_t> shp;
  CUdeviceptr dev_ptr_fake;
  if (is_dynamic_) {
    runtimeargs.reserve(arg_size_vec_.size());
    dev_addrs.reserve(arg_size_vec_.size());
    shp.reserve(arg_size_vec_.size());
    size_t inum = 0;
    size_t onum = 0;
    for (size_t idx = 0; idx < arg_size_vec_.size(); idx++) {
      if (arg_size_vec_[idx] == kRemove) {
        runtimeargs.push_back(reinterpret_cast<void *>(&dev_ptr_fake));
      } else if (arg_size_vec_[idx] == kKeep) {
        if (inum < inputs.size()) {
          runtimeargs.push_back(reinterpret_cast<void *>(&dev_addrs.emplace_back(inputs[inum]->device_ptr())));
          inum++;
        } else if (onum < outputs.size()) {
          runtimeargs.push_back(reinterpret_cast<void *>(&dev_addrs.emplace_back(outputs[onum]->device_ptr())));
          onum++;
        }
      } else {
        size_t arg_size = static_cast<size_t>(arg_size_vec_[idx]);
        runtimeargs.push_back(reinterpret_cast<void *>(&shp.emplace_back(arg_size)));
      }
    }
  } else {
    runtimeargs.reserve(inputs.size() + outputs.size() + workspace.size());
    dev_addrs.reserve(inputs.size() + outputs.size() + workspace.size());
    (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtimeargs),
                         [&dev_addrs](const KernelTensor *input) {
                           return reinterpret_cast<void *>(&dev_addrs.emplace_back(input->device_ptr()));
                         });
    (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtimeargs),
                         [&dev_addrs](const KernelTensor *output) {
                           return reinterpret_cast<void *>(&dev_addrs.emplace_back(output->device_ptr()));
                         });
    if (!workspace.empty()) {
      (void)std::transform(std::begin(workspace), std::end(workspace), std::back_inserter(runtimeargs),
                           [&dev_addrs](const KernelTensor *ws) {
                             return reinterpret_cast<void *>(&dev_addrs.emplace_back(ws->device_ptr()));
                           });
    }
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
  MS_LOG(INFO) << "End Launch for " << kernel_name_;
  return true;
}
}  // namespace kernel
}  // namespace mindspore
