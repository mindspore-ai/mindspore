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

#include "plugin/device/gpu/kernel/dynamic_akg/dynamic_akg_gpu_kernel_mod.h"
#include <fstream>
#include <algorithm>
#include <map>
#include "nlohmann/json.hpp"
#include "utils/ms_utils.h"
#include "kernel/framework_utils.h"
#include "mindspore/ccsrc/include/common/debug/common.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"

namespace mindspore {
namespace kernel {
using std::fstream;
using std::string;
using std::vector;

constexpr auto kMappingUpdated = "updated";
constexpr auto kBlockIdxX = "blockIdx.x";
constexpr auto kBlockIdxY = "blockIdx.y";
constexpr auto kBlockIdxZ = "blockIdx.z";
constexpr auto kThreadIdxX = "threadIdx.x";
constexpr auto kThreadIdxY = "threadIdx.y";
constexpr auto kThreadIdxZ = "threadIdx.z";
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

void PrintCuError(CUresult result, const string error_msg, const string kernel_name) {
  if (result != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(result, &msg);
    MS_LOG(ERROR) << error_msg << " Kernel name: << " << kernel_name << ". Error message: " << msg;
  }
}

DynamicAkgGpuKernelManagerPtr DynamicAkgGpuKernelMod::kernel_manager_ = std::make_shared<DynamicAkgGpuKernelManager>();
DynamicAkgGpuKernelManager::DynamicAkgGpuKernelManager() {}

CUresult DynamicAkgGpuKernelManager::GetFunction(const KernelPackPtr &kernel_pack, bool force_reload,
                                                 vector<uint32_t> *thread_info, CUfunction *func,
                                                 std::unordered_map<std::string, int64_t> map_info,
                                                 const std::string kernel_name) {
  if (kernel_pack->GetJson() == nullptr || kernel_pack->GetJson()->contents == nullptr ||
      kernel_pack->GetKernel() == nullptr || kernel_pack->GetKernel()->contents == nullptr) {
    MS_LOG(ERROR) << "GPU:Invalid kernel pack, json or kernel is nullptr.";
    return CUDA_ERROR_INVALID_IMAGE;
  }
  string fn = kernel_name;
  if (!force_reload) {
    auto iter = infotable_.find(fn);
    if (iter != infotable_.end()) {
      auto kernelmeta = iter->second;
      *thread_info = kernelmeta->thread_info_;
      *func = kernelmeta->func_addr_;
      return CUDA_SUCCESS;
    }
  }

  thread_info->emplace_back(map_info[kBlockIdxX]);
  thread_info->emplace_back(map_info[kBlockIdxY]);
  thread_info->emplace_back(map_info[kBlockIdxZ]);
  thread_info->emplace_back(map_info[kThreadIdxX]);
  thread_info->emplace_back(map_info[kThreadIdxY]);
  thread_info->emplace_back(map_info[kThreadIdxZ]);

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

void DynamicAkgGpuKernelMod::InitMappingInfo() {
  map_info_[kBlockIdxX] = -1;
  map_info_[kBlockIdxY] = -1;
  map_info_[kBlockIdxZ] = -1;
  map_info_[kThreadIdxX] = -1;
  map_info_[kThreadIdxY] = -1;
  map_info_[kThreadIdxZ] = -1;
  map_info_[kMappingUpdated] = 0;
}

void DynamicAkgGpuKernelMod::UpdateMappingInfo() {
  if (kernel_pack_ == nullptr) {
    MS_LOG(ERROR) << "GPU:Invalid kernel pack, json or kernel is nullptr.";
    return;
  }
  auto js = kernel_pack_->GetJson();
  if (js == nullptr) {
    MS_LOG(ERROR) << "GPU:Invalid kernel pack, json or kernel is nullptr.";
    return;
  }
  auto parsed_js = nlohmann::json::parse(js->contents, js->contents + js->len);
  if (is_dynamic_) {
    MS_LOG(WARNING) << "Dynamic shape not supported.";
    return;
  } else {
    map_info_[kBlockIdxX] = parsed_js[kBlockIdxX];
    map_info_[kBlockIdxY] = parsed_js[kBlockIdxY];
    map_info_[kBlockIdxZ] = parsed_js[kBlockIdxZ];
    map_info_[kThreadIdxX] = parsed_js[kThreadIdxX];
    map_info_[kThreadIdxY] = parsed_js[kThreadIdxY];
    map_info_[kThreadIdxZ] = parsed_js[kThreadIdxZ];
  }
  map_info_[kMappingUpdated] = 1;
}

DynamicAkgGpuKernelMod::DynamicAkgGpuKernelMod(const KernelPackPtr &kernel_pack) : kernel_pack_(kernel_pack) {
  if (kernel_pack != nullptr) {
    auto js = kernel_pack->GetJson();
    if (js != nullptr) {
      auto parsed_js = nlohmann::json::parse(js->contents, js->contents + js->len);
      kernel_name_ = parsed_js["kernelName"];
    }
  }
  InitMappingInfo();
}

int DynamicAkgGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  MS_LOG(DEBUG) << "Start resize for DynamicAkgGpuKernelMod.";
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  UpdateShapeList(inputs, outputs);
  UpdateMappingInfo();
  MS_LOG(DEBUG) << "Done resize for DynamicAkgGpuKernelMod.";
  return ret;
}

void DynamicAkgGpuKernelMod::UpdateShapeList(const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  ndims_.clear();
  shape_list_.clear();
  for (size_t i = 0; i < inputs.size(); i++) {
    auto in_shape = inputs[i]->GetShapeVector();
    (void)shape_list_.emplace_back(in_shape);
    ndims_.push_back(SizeToInt(in_shape.size()));
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    auto out_shape = outputs[i]->GetShapeVector();
    (void)shape_list_.emplace_back(out_shape);
    ndims_.push_back(SizeToInt(out_shape.size()));
  }
}

std::vector<std::vector<int64_t>> DynamicAkgGpuKernelMod::GetArgSizeVec() {
  auto max_length_iter =
    std::max_element(shape_list_.begin(), shape_list_.end(),
                     [](const std::vector<int64_t> &a, const std::vector<int64_t> &b) { return a.size() < b.size(); });
  size_t max_length = max_length_iter->size();
  std::vector<std::vector<int64_t>> arg_size_vec;
  arg_size_vec.reserve(ndims_.size());
  for (size_t i = 0; i < ndims_.size(); i++) {
    std::vector<int64_t> arg_size;
    arg_size.push_back(0);
    arg_size.insert(arg_size.end(), shape_list_[i].begin(), shape_list_[i].end());
    std::vector<int64_t> strides_(ndims_[i], 1);
    for (int j = SizeToInt(ndims_[i]) - 2; j >= 0; j--) {
      strides_[j] = strides_[j + 1] * shape_list_[i][j + 1];
    }
    const size_t kTwo = 2;
    (void)arg_size.insert(arg_size.end(), strides_.begin(), strides_.end());
    (void)arg_size.insert(arg_size.end(), kTwo * (max_length - shape_list_[i].size()), 0);
    arg_size_vec.push_back(arg_size);
  }
  return arg_size_vec;
}

bool DynamicAkgGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == 0) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr. Kernel name: " << kernel_name_;
    return false;
  }
  if (kernel_pack_ == nullptr) {
    MS_LOG(ERROR) << "kernel pack should not be nullptr. Kernel name: " << kernel_name_;
    return false;
  }
  if (is_dynamic_) {
    MS_EXCEPTION(RuntimeError) << "Dynamic shape not supported.";
  }
  CUresult result;
  if (kernel_addr_ == nullptr) {
    if (!map_info_[kMappingUpdated]) {
      UpdateMappingInfo();
    }
    result = kernel_manager_->GetFunction(kernel_pack_, false, &thread_info_, &kernel_addr_, map_info_, kernel_name_);
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
  if (is_dynamic_) {
    // calculate shape info: [0, dims, strides] and update workspace
    // std::vector<std::vector<int64_t>> arg_size_vec = GetArgSizeVec();
    // create an m x (1 + 2 x n) array for M tensors and N dims while RESIZE
    // alloc and copy to device, append address to runtimeargs
    MS_LOG(WARNING) << "Dynamic shape not supported.";
  }
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
