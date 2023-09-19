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
#include "plugin/device/gpu/kernel/dynamic_akg/replace_ptx_utils.h"

namespace mindspore {
namespace kernel {
using std::fstream;
using std::string;
using std::unordered_map;
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
constexpr auto kMappingUpdated = "updated";
constexpr auto kBlockIdxX = "blockIdx.x";
constexpr auto kBlockIdxY = "blockIdx.y";
constexpr auto kBlockIdxZ = "blockIdx.z";
constexpr auto kThreadIdxX = "threadIdx.x";
constexpr auto kThreadIdxY = "threadIdx.y";
constexpr auto kThreadIdxZ = "threadIdx.z";
constexpr auto kHostShapes = "hostShapes";
constexpr auto kDeviceShapes = "deviceShapes";

DynamicAkgGpuKernelManagerPtr DynamicAkgGpuKernelMod::kernel_manager_ = std::make_shared<DynamicAkgGpuKernelManager>();
DynamicAkgGpuKernelManager::DynamicAkgGpuKernelManager() {}

CUresult DynamicAkgGpuKernelManager::GetCUResult(const char *kernel_content, bool force_reload,
                                                 vector<uint32_t> *thread_info, CUfunction *func,
                                                 const string kernel_name, unordered_map<string, int64_t> map_info) {
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
  MS_LOG(INFO) << kernel_name << ", thread_info = " << *thread_info;

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

  CUresult result = cuModuleLoadDataEx(&module, kernel_content, 1, options, values);
  if (result != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(result, &msg);
    MS_LOG(INFO) << kernel_content;
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
                                                 const string kernel_name, unordered_map<string, int64_t> map_info) {
  if (kernel_pack->GetJson() == nullptr || kernel_pack->GetJson()->contents == nullptr ||
      kernel_pack->GetKernel() == nullptr || kernel_pack->GetKernel()->contents == nullptr) {
    MS_LOG(ERROR) << "Invalid kernel pack, json or kernel is nullptr of kernel : " << kernel_name << ".\n";
    return CUDA_ERROR_INVALID_IMAGE;
  }
  return GetCUResult(&kernel_pack->GetKernel()->contents[0], force_reload, thread_info, func, kernel_name, map_info);
}

CUresult DynamicAkgGpuKernelManager::GetFunctionFromStr(const string ptx_str, bool force_reload,
                                                        vector<uint32_t> *thread_info, CUfunction *func,
                                                        const string kernel_name,
                                                        unordered_map<string, int64_t> map_info) {
  return GetCUResult(ptx_str.c_str(), force_reload, thread_info, func, kernel_name, map_info);
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

void DynamicAkgGpuKernelMod::UpdateShapeList(const vector<KernelTensorPtr> &inputs,
                                             const vector<KernelTensorPtr> &outputs) {
  shape_list_.clear();
  for (size_t i = 0; i < inputs.size(); i++) {
    auto in_shape = inputs[i]->GetShapeVector();
    (void)shape_list_.emplace_back(in_shape);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    auto out_shape = outputs[i]->GetShapeVector();
    (void)shape_list_.emplace_back(out_shape);
  }
}

void DynamicAkgGpuKernelMod::GetDeviceShape() {
  // get the real number for each symbol shape
  for (size_t i = 0; i < shape_list_.size(); i++) {
    for (size_t j = 0; j < shape_list_[i].size(); j++) {
      if (i >= parsed_js_[kHostShapes].size() || j >= parsed_js_[kHostShapes][i].size()) {
        MS_EXCEPTION(RuntimeError) << "Host shapes during runtime does not match AkgV2 kernel: " << kernel_name_ << ".";
      }
      string shape_str = parsed_js_[kHostShapes][i][j];
      if (!std::all_of(shape_str.begin(), shape_str.end(), [](char c) { return std::isdigit(c); })) {
        symbol_map_[shape_str] = shape_list_[i][j];
      }
    }
  }
  // get device shape list
  for (size_t i = 0; i < parsed_js_[kDeviceShapes].size(); i++) {
    vector<int64_t> device_tensor_shape;
    for (size_t j = 0; j < parsed_js_[kDeviceShapes][i].size(); j++) {
      string device_shape_str = parsed_js_[kDeviceShapes][i][j];
      if (std::all_of(device_shape_str.begin(), device_shape_str.end(), [](char c) { return std::isdigit(c); })) {
        (void)device_tensor_shape.emplace_back(std::stoi(device_shape_str));
      } else {
        (void)device_tensor_shape.emplace_back(symbol_map_[device_shape_str]);
      }
    }
    (void)device_shape_list_.emplace_back(device_tensor_shape);
  }
}

void DynamicAkgGpuKernelMod::GetDeviceArgSizeVec() {
  device_shape_list_.clear();
  arg_size_vec_.clear();
  GetDeviceShape();
  for (size_t i = 0; i < device_shape_list_.size(); i++) {
    vector<int64_t> arg_size;
    arg_size.push_back(0);
    auto device_tensor_shape = device_shape_list_[i];
    arg_size.insert(arg_size.end(), device_tensor_shape.begin(), device_tensor_shape.end());
    vector<int64_t> strides(device_tensor_shape.size(), 1);
    for (int j = SizeToInt(device_tensor_shape.size()) - 2; j >= 0; j--) {
      strides[j] = strides[j + 1] * device_tensor_shape[j + 1];
    }
    (void)arg_size.insert(arg_size.end(), strides.begin(), strides.end());
    arg_size_vec_.push_back(arg_size);
  }
}

void DynamicAkgGpuKernelMod::UpdateMappingInfo() {
  if (is_dynamic_) {
    // compute map arg substituting symbol shape value
    vector<string> map_arg_list = {kBlockIdxX, kBlockIdxY, kBlockIdxZ, kThreadIdxX, kThreadIdxY, kThreadIdxZ};
    for (auto map_arg : map_arg_list) {
      if (parsed_js_[map_arg].is_number()) {
        map_info_[map_arg] = parsed_js_[map_arg];
      } else if (parsed_js_[map_arg].is_array() && parsed_js_[map_arg].size() == 2 &&
                 parsed_js_[map_arg][0].is_string() && parsed_js_[map_arg][1].is_number()) {
        auto dim_size_float = static_cast<float>(symbol_map_[parsed_js_[map_arg][0]]);
        auto tile_size_float = static_cast<float>(parsed_js_[map_arg][1]);
        map_info_[map_arg] = static_cast<int64_t>(std::ceil(dim_size_float / tile_size_float));
      } else {
        MS_LOG(ERROR) << "Mapping info format error.";
        return;
      }
    }
  } else {
    map_info_[kBlockIdxX] = parsed_js_[kBlockIdxX];
    map_info_[kBlockIdxY] = parsed_js_[kBlockIdxY];
    map_info_[kBlockIdxZ] = parsed_js_[kBlockIdxZ];
    map_info_[kThreadIdxX] = parsed_js_[kThreadIdxX];
    map_info_[kThreadIdxY] = parsed_js_[kThreadIdxY];
    map_info_[kThreadIdxZ] = parsed_js_[kThreadIdxZ];
  }
  map_info_[kMappingUpdated] = 1;
}

void DynamicAkgGpuKernelMod::ReplacePTX() {
  if (kernel_pack_->GetKernel() == nullptr || kernel_pack_->GetKernel()->contents == nullptr) {
    MS_EXCEPTION(RuntimeError) << "Invalid kernel pack, kernel is nullptr of kernel :" << kernel_name_;
    return;
  }
  auto kernel_contents = kernel_pack_->GetKernel()->contents;
  string original_ptx(kernel_contents);
  replaced_ptx_ = ReplacePTXFunction(original_ptx, arg_size_vec_, kernel_name_);
}

DynamicAkgGpuKernelMod::DynamicAkgGpuKernelMod(const KernelPackPtr &kernel_pack) : kernel_pack_(kernel_pack) {
  if (kernel_pack != nullptr) {
    auto js = kernel_pack->GetJson();
    if (js != nullptr) {
      parsed_js_ = nlohmann::json::parse(js->contents, js->contents + js->len);
      kernel_name_ = parsed_js_["kernelName"];
    }
  }
  InitMappingInfo();
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

int DynamicAkgGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const vector<KernelTensorPtr> &inputs,
                                   const vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  MS_LOG(DEBUG) << "Start resize for DynamicAkgGpuKernelMod for " << kernel_name_;
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  CheckJsonParsed();
  UpdateShapeList(inputs, outputs);
  GetDeviceArgSizeVec();
  UpdateMappingInfo();
  ReplacePTX();
  MS_LOG(DEBUG) << "Done resize for DynamicAkgGpuKernelMod for " << kernel_name_;
  return ret;
}

bool DynamicAkgGpuKernelMod::Launch(const vector<AddressPtr> &inputs, const vector<AddressPtr> &workspace,
                                    const vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == 0) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr. Kernel name: " << kernel_name_;
    return false;
  }
  if (kernel_pack_ == nullptr) {
    MS_LOG(ERROR) << "kernel pack should not be nullptr. Kernel name: " << kernel_name_;
    return false;
  }
  MS_LOG(INFO) << kernel_name_ << ", Start Launch.";
  CUresult result;
  if (is_dynamic_) {
    // for dynamic shape, map info is updated during Resize
    if (!map_info_[kMappingUpdated]) {
      MS_LOG(ERROR) << "For dynamic shape " << kernel_name_ << ", gpu mapping config must be updated before launch";
      return false;
    }
    for (size_t i = 0; i < inputs.size(); i++) {
      MS_LOG(INFO) << kernel_name_ << ", input[" << i << "], shape_list = " << shape_list_[i]
                   << ", device_shape_list = " << device_shape_list_[i] << ", arg_size_vec = " << arg_size_vec_[i];
    }
    for (size_t j = 0; j < outputs.size(); j++) {
      auto i = inputs.size() + j;
      MS_LOG(INFO) << kernel_name_ << ", output[" << j << "], shape_list = " << shape_list_[i]
                   << ", device_shape_list = " << device_shape_list_[i] << ", arg_size_vec = " << arg_size_vec_[i];
    }
    thread_info_.clear();
    result =
      kernel_manager_->GetFunctionFromStr(replaced_ptx_, true, &thread_info_, &kernel_addr_, kernel_name_, map_info_);
    if (result != CUDA_SUCCESS) {
      const char *msg = nullptr;
      cuGetErrorName(result, &msg);
      MS_LOG(ERROR) << "Get function " << kernel_name_ << " failed. Error message: " << msg;
      return false;
    }
  } else if (kernel_addr_ == nullptr) {
    CheckJsonParsed();
    UpdateMappingInfo();
    if (!map_info_[kMappingUpdated]) {
      MS_LOG(ERROR) << "For " << kernel_name_ << ", gpu mapping config must be updated before launch";
      return false;
    }
    result = kernel_manager_->GetFunction(kernel_pack_, false, &thread_info_, &kernel_addr_, kernel_name_, map_info_);
    if (result != CUDA_SUCCESS) {
      const char *msg = nullptr;
      cuGetErrorName(result, &msg);
      MS_LOG(ERROR) << "Get function " << kernel_name_ << " failed. Error message: " << msg;
      return false;
    }
  }

  vector<void *> runtimeargs;
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
  MS_LOG(INFO) << kernel_name_ << ", End Launch.";
  return true;
}
}  // namespace kernel
}  // namespace mindspore
