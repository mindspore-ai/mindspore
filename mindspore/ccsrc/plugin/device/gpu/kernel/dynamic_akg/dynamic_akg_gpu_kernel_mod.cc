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
constexpr auto kBlockIdxX = "blockIdx.x";
constexpr auto kBlockIdxY = "blockIdx.y";
constexpr auto kBlockIdxZ = "blockIdx.z";
constexpr auto kThreadIdxX = "threadIdx.x";
constexpr auto kThreadIdxY = "threadIdx.y";
constexpr auto kThreadIdxZ = "threadIdx.z";
constexpr auto kHostShapes = "hostShapes";
constexpr auto kDeviceShapes = "deviceShapes";
constexpr auto kRemove = -100000;
constexpr auto kKeep = -99999;

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

void DynamicAkgGpuKernelMod::InitJsonShapeInformation() {
  // Initialize device shape list using -1 for unknown dims
  // Record map <device_loc, symbol>
  unordered_map<std::pair<size_t, size_t>, string, PairHash> device_loc_map;
  for (size_t i = 0; i < parsed_js_[kDeviceShapes].size(); i++) {
    vector<int64_t> device_tensor_shape;
    for (size_t j = 0; j < parsed_js_[kDeviceShapes][i].size(); j++) {
      string device_shape_str = parsed_js_[kDeviceShapes][i][j];
      if (std::all_of(device_shape_str.begin(), device_shape_str.end(), [](char c) { return std::isdigit(c); })) {
        (void)device_tensor_shape.emplace_back(std::stoi(device_shape_str));
      } else {
        (void)device_tensor_shape.emplace_back(-1);
        auto device_loc = std::make_pair(i, j);
        device_loc_map[device_loc] = device_shape_str;
      }
    }
    (void)device_shape_list_.emplace_back(device_tensor_shape);
  }
  // Record map <symbol, host_loc>
  std::unordered_map<std::string, std::pair<size_t, size_t>> host_loc_map;
  for (size_t i = 0; i < parsed_js_[kHostShapes].size(); i++) {
    for (size_t j = 0; j < parsed_js_[kHostShapes][i].size(); j++) {
      string shape_str = parsed_js_[kHostShapes][i][j];
      if ((!std::all_of(shape_str.begin(), shape_str.end(), [](char c) { return std::isdigit(c); })) &&
          (host_loc_map.find(shape_str) == host_loc_map.end())) {
        host_loc_map[shape_str] = std::make_pair(i, j);
      }
    }
  }
  // Get map <device_loc, host_loc>
  for (auto item : device_loc_map) {
    device_host_shape_loc_[item.first] = host_loc_map[item.second];
  }
  // Initialize mapping info as a vector. Only store dividend number for unknown mapping args
  // Record map <unknown map ard id, host_loc>
  init_mapping_.clear();
  vector<string> map_arg_list = {kBlockIdxX, kBlockIdxY, kBlockIdxZ, kThreadIdxX, kThreadIdxY, kThreadIdxZ};
  for (size_t i = 0; i < map_arg_list.size(); i++) {
    auto map_arg = map_arg_list[i];
    if (parsed_js_[map_arg].is_number()) {
      (void)init_mapping_.emplace_back(parsed_js_[map_arg]);
    } else if (parsed_js_[map_arg].is_array() && parsed_js_[map_arg].size() == 2 &&
               parsed_js_[map_arg][0].is_string() && parsed_js_[map_arg][1].is_number()) {
      string divisor_symbol = parsed_js_[map_arg][0];
      auto dividend = parsed_js_[map_arg][1];
      (void)init_mapping_.emplace_back(static_cast<uint32_t>(dividend));
      unknown_map_loc_[i] = host_loc_map[divisor_symbol];
    } else {
      MS_EXCEPTION(RuntimeError) << "Mapping info format error.";
      return;
    }
  }
  json_shape_updated_ = true;
  MS_LOG(INFO) << "Done InitJsonShapeInformation for " << kernel_name_;
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
  MS_LOG(INFO) << "Done UpdateShapeList for " << kernel_name_;
}

void DynamicAkgGpuKernelMod::GetDeviceArgSizeVec(const size_t inputs_num) {
  arg_size_vec_.clear();
  std::vector<std::vector<int64_t>> device_shape(device_shape_list_);
  // Update each unknown value in device_shape_list to get real device shape
  for (auto item : device_host_shape_loc_) {
    auto device_loc = item.first;
    auto host_loc = item.second;
    device_shape[device_loc.first][device_loc.second] = shape_list_[host_loc.first][host_loc.second];
  }

  for (size_t i = 0; i < device_shape.size(); i++) {
    if (i < inputs_num) {
      MS_LOG(DEBUG) << "For " << kernel_name_ << ", input[" << i << "]: host_shape = " << shape_list_[i]
                    << ", device_shape = " << device_shape[i];
    } else {
      MS_LOG(DEBUG) << "For " << kernel_name_ << ", output[" << i - inputs_num << "]: host_shape = " << shape_list_[i]
                    << ", device_shape = " << device_shape[i];
    }
    arg_size_vec_.push_back(kRemove);  // useless in memref
    arg_size_vec_.push_back(kKeep);    // data ptr
    arg_size_vec_.push_back(0);        // offset
    auto device_tensor_shape = device_shape[i];
    for (auto item : device_tensor_shape) {
      if (item <= 0) {
        MS_EXCEPTION(RuntimeError) << "Shape still have negative value for kernel: " << kernel_name_
                                   << "with host shape[" << i << "] = " << shape_list_[i] << ", device_tensor_shape["
                                   << i << "] = " << device_tensor_shape;
      }
      arg_size_vec_.push_back(item);
    }
    vector<int64_t> strides(device_tensor_shape.size(), 1);
    for (int j = SizeToInt(device_tensor_shape.size()) - 2; j >= 0; j--) {
      strides[j] = strides[j + 1] * device_tensor_shape[j + 1];
    }
    for (auto item : strides) {
      arg_size_vec_.push_back(item);
    }
  }
  MS_LOG(DEBUG) << "For " << kernel_name_ << ", arg_size_vec = " << arg_size_vec_;
  MS_LOG(INFO) << "Done GetDeviceArgSizeVec for " << kernel_name_;
}

void DynamicAkgGpuKernelMod::UpdateDynamicShapeMappingInfo() {
  thread_info_ = init_mapping_;
  for (auto item : unknown_map_loc_) {
    auto thread_info_id = item.first;
    if (thread_info_id >= thread_info_.size()) {
      MS_EXCEPTION(RuntimeError) << "Unknown thread arg index should not exceed thread_info length, "
                                 << "which is 6 (Grid.X/Y/Z, Block.X/Y/Z).";
    }
    auto host_loc = item.second;
    auto dim_size = shape_list_[host_loc.first][host_loc.second];
    auto tile_size = thread_info_[thread_info_id];
    auto dim_size_float = static_cast<float>(dim_size);
    auto tile_size_float = static_cast<float>(tile_size);
    thread_info_[thread_info_id] = static_cast<uint32_t>(std::ceil(dim_size_float / tile_size_float));
  }

  MS_LOG(DEBUG) << "For " << kernel_name_ << ",  thread_info = " << thread_info_;
  MS_LOG(INFO) << "Done UpdateDynamicShapeMappingInfo for " << kernel_name_;
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

int DynamicAkgGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const vector<KernelTensorPtr> &inputs,
                                   const vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  CheckJsonParsed();
  if (!json_shape_updated_) {
    InitJsonShapeInformation();
  }
  UpdateShapeList(inputs, outputs);
  GetDeviceArgSizeVec(inputs.size());
  UpdateDynamicShapeMappingInfo();
  MS_LOG(INFO) << "Done resize for DynamicAkgGpuKernelMod for " << kernel_name_;
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
  MS_LOG(INFO) << "Start Launch for " << kernel_name_;
  CUresult result;
  if (kernel_addr_ == nullptr) {
    if (!is_dynamic_) {
      CheckJsonParsed();
      UpdateStaticShapeMappingInfo();
    }
    if (thread_info_.size() != 6 ||
        (std::any_of(thread_info_.begin(), thread_info_.end(), [](uint32_t t) { return t <= 0; }))) {
      MS_LOG(ERROR) << "For " << kernel_name_ << ", gpu mapping config must be updated to 6 positive numbers before "
                    << "launch, but got thread_info = " << thread_info_;
      return false;
    }
    result = kernel_manager_->GetFunction(kernel_pack_, false, &thread_info_, &kernel_addr_, kernel_name_);
    if (result != CUDA_SUCCESS) {
      const char *msg = nullptr;
      cuGetErrorName(result, &msg);
      MS_LOG(ERROR) << "Get function " << kernel_name_ << " failed. Error message: " << msg;
      return false;
    }
  }

  vector<void *> runtimeargs;
  if (is_dynamic_) {
    size_t inum = 0;
    size_t onum = 0;
    CUdeviceptr dev_ptr_fake;
    for (size_t idx = 0; idx < arg_size_vec_.size(); idx++) {
      if (arg_size_vec_[idx] == kRemove) {
        runtimeargs.push_back(reinterpret_cast<void *>(&dev_ptr_fake));
      } else if (arg_size_vec_[idx] == kKeep) {
        if (inum < inputs.size()) {
          runtimeargs.push_back(reinterpret_cast<void *>(&(inputs[inum]->addr)));
          inum++;
        } else if (onum < outputs.size()) {
          runtimeargs.push_back(reinterpret_cast<void *>(&(outputs[onum]->addr)));
          onum++;
        }
      } else {
        size_t *value_ptr = new size_t(arg_size_vec_[idx]);
        runtimeargs.push_back(value_ptr);
      }
    }
  } else {
    runtimeargs.reserve(inputs.size() + outputs.size() + workspace.size());
    (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtimeargs),
                         [](const AddressPtr &input) { return reinterpret_cast<void *>(&(input->addr)); });
    (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtimeargs),
                         [](const AddressPtr &output) { return reinterpret_cast<void *>(&(output->addr)); });
    if (!workspace.empty()) {
      (void)std::transform(std::begin(workspace), std::end(workspace), std::back_inserter(runtimeargs),
                           [](const AddressPtr &addr) { return reinterpret_cast<void *>(&(addr->addr)); });
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
