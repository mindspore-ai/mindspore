/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "kernel/graph_kernel/kernel_packet/kernel_packet_kernel_mod.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "include/backend/anf_runtime_algorithm.h"
#include "ir/anf.h"
#include "kernel/common_utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/utils/convert_utils.h"
#include "mindspore/core/symbolic_shape/utils.h"
#include "mindspore/core/symbolic_shape/symbol_engine.h"
#include "abstract/abstract_value.h"
#include "kernel/graph_kernel/kernel_packet/kernel_packet_infer_functor.h"

namespace mindspore::kernel {
bool KernelPacketInitializer::InitKernel(const CNodePtr &real_node, const KernelModPtr &real_kernel_mod,
                                         KernelPacketKernelMod *packet_kernel_mod, KernelPacketInfer *infer) {
  MS_EXCEPTION_IF_NULL(real_node);
  packet_kernel_mod->real_node_debug_str_ = real_node->DebugString();
  FuncGraphPtr func_graph = real_node->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Empty func_graph of " << packet_kernel_mod->real_node_debug_str_;
    return false;
  }
  auto symbol_engine = func_graph->symbol_engine();
  if (symbol_engine == nullptr) {
    MS_LOG(ERROR) << "Empty symbol engine of func_graph of " << packet_kernel_mod->real_node_debug_str_;
    return false;
  }
  size_t input_tensor_num = common::AnfAlgo::GetInputTensorNum(real_node);
  packet_kernel_mod->inputs_cache_.reserve(input_tensor_num);
  infer->SetInnerInputNum(input_tensor_num);
  packet_kernel_mod->host_value_cache_.clear();
  packet_kernel_mod->host_value_cache_.resize(input_tensor_num);
  auto &outer_inputs = func_graph->parameters();
  for (size_t i = 0; i < input_tensor_num; i++) {
    auto prev_node = real_node->input(i + 1);
    auto abs = prev_node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    MS_LOG(DEBUG) << "The realnode " << real_node->DebugString() << " input[" << i << "] is "
                  << prev_node->DebugString();
    auto value = abs->GetValue();
    auto iter = std::find(outer_inputs.begin(), outer_inputs.end(), prev_node);
    if (iter != outer_inputs.end()) {
      packet_kernel_mod->input_map_[i] = static_cast<size_t>(iter - outer_inputs.begin());
    } else if (value == nullptr || value->isa<ValueAny>()) {
      infer->SetInnerInput(i, prev_node->abstract());
    } else {
      MS_LOG(DEBUG) << "The input " << i << " is a const value: " << value->ToString();
    }
    auto kernel_tensor = std::make_shared<KernelTensor>(abs->GetShape(), abs->GetType(), value);
    (void)packet_kernel_mod->inputs_cache_.emplace_back(std::move(kernel_tensor));
  }
  packet_kernel_mod->real_kernel_mod_ = real_kernel_mod;
  return true;
}

void KernelPacketKernelMod::AllocWorkspace(size_t i, size_t data_size) {
  MS_LOG(DEBUG) << "Allocate " << data_size << " bytes workspace for input " << i;
  if (data_size == 0) {
    data_size = 1;
  }
  input_workspace_map_[i] = workspace_size_list_.size();
  workspace_size_list_.push_back(data_size);
}

int KernelPacketKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  MS_LOG(DEBUG) << "Resize begin: " << kernel_name_;
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_workspace_map_.clear();
  auto inner_input_num = inputs_cache_.size();
  std::vector<KernelTensor *> inner_inputs(inner_input_num, nullptr);
  for (size_t i = 0; i < inner_input_num; ++i) {
    if (auto iter = input_map_.find(i); iter != input_map_.end()) {
      MS_LOG(DEBUG) << "Inner input " << i << " use outer input " << iter->second;
      inner_inputs[i] = inputs[iter->second];
      continue;
    }
    if (host_value_cache_[i] != nullptr) {
      auto ori = inputs_cache_[i];
      BaseShapePtr shape = ori->GetShape();
      if (shape->IsDynamic()) {
        shape = host_value_cache_[i]->ToAbstract()->GetShape();
      }
      MS_LOG(DEBUG) << "Inner input " << i << " is host value: " << host_value_cache_[i]->ToString()
                    << ". Its shape is " << shape->ToString() << ", the type is " << ori->GetType();
      inputs_cache_[i] = std::make_shared<KernelTensor>(shape, ori->GetType(), host_value_cache_[i]);
      inner_inputs[i] = inputs_cache_[i].get();
    } else {
      inner_inputs[i] = inputs_cache_[i].get();
      MS_LOG(DEBUG) << "Inner input " << i << " of " << real_node_debug_str_ << " is const value.";
    }
    AllocWorkspace(i, inner_inputs[i]->size());
  }

  auto res = real_kernel_mod_->Resize(inner_inputs, outputs);
  MS_LOG(DEBUG) << "Inner kernel resize finished: " << real_node_debug_str_;
  if (res != KRET_OK) {
    return res;
  }
  const auto &workspace = real_kernel_mod_->GetWorkspaceSizeList();
  if (!workspace.empty()) {
    MS_LOG(DEBUG) << "Inner kernel workspaces size: " << workspace.size();
    workspace_size_list_.reserve(workspace.size() + workspace_size_list_.size());
    // Inner kernel's workspace is behind shape workspace
    (void)workspace_size_list_.insert(workspace_size_list_.end(), workspace.begin(), workspace.end());
  }

  // first call for KernelTensor::GetValuePtr  will convert the ValuePtr to void*.
  // call this interface in Resize can reduce the launch time
  host_data_cache_.clear();
  host_data_cache_.resize(inner_input_num, nullptr);
  std::vector<size_t> ignore_idx = real_kernel_mod_->GetLaunchIgnoredInputAddressIdx();
  for (size_t i = 0; i < inner_input_num; i++) {
    if (input_map_.count(i) != 0) {
      continue;
    }
    if (inner_inputs[i]->size() > 0 && std::find(ignore_idx.begin(), ignore_idx.end(), i) == ignore_idx.end()) {
      host_data_cache_[i] = inner_inputs[i]->GetValuePtr();
    }
  }

  MS_LOG(DEBUG) << "Resize end: " << kernel_name_;
  return KRET_OK;
}

bool KernelPacketKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspaces,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(DEBUG) << "Launch begin: " << kernel_name_;
  auto [inner_inputs, inner_workspaces] = GetLaunchArgs(inputs, workspaces, stream_ptr);
  auto res = real_kernel_mod_->Launch(inner_inputs, inner_workspaces, outputs, stream_ptr);
  MS_LOG(DEBUG) << "Finish inner kernel launch: " << real_node_debug_str_;
  if (!res) {
    MS_LOG(ERROR) << "Launch kernel: " << real_node_debug_str_ << " failed.";
    return false;
  }
  MS_LOG(DEBUG) << "Launch end: " << kernel_name_;
  return true;
}

std::vector<KernelAttr> KernelPacketKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
  return support_list;
}

KernelPacketKernelMod::AddressArgs KernelPacketKernelMod::GetLaunchArgs(const std::vector<KernelTensor *> &inputs,
                                                                        const std::vector<KernelTensor *> &workspaces,
                                                                        void *stream_ptr) {
  std::vector<KernelTensor *> res_inputs;
  res_inputs.resize(inputs_cache_.size(), nullptr);
  for (size_t i = 0; i < inputs_cache_.size(); i++) {
    if (auto iter = input_map_.find(i); iter != input_map_.end()) {
      auto j = iter->second;
      MS_LOG(DEBUG) << "Inner input " << i << " -> outer input " << j;
      res_inputs[i] = inputs[j];
    } else if (auto iter = input_workspace_map_.find(i); iter != input_workspace_map_.end()) {
      auto j = iter->second;
      MS_LOG(DEBUG) << "Inner input " << i << " -> workspace " << j;
      res_inputs[i] = inputs_cache_[i].get();
      // set the device_ptr of workspaces to res_input
      res_inputs[i]->set_pointer_ref_count(workspaces[j]->pointer_ref_count());
      // copy host data to device
      if (host_data_cache_[i] != nullptr) {
        MS_LOG(DEBUG) << "Copy input " << i << " from host to device. device_ptr: " << res_inputs[i]->device_ptr()
                      << ", size: " << res_inputs[i]->size();
        CopyHostToDevice(res_inputs[i]->device_ptr(), host_data_cache_[i], res_inputs[i]->size(), stream_ptr);
      }
    } else {
      MS_LOG(DEBUG) << "Inner input " << i << " is not found in input_map and input_workspace_map.";
      res_inputs[i] = inputs_cache_[i].get();
    }
  }
  MS_LOG(DEBUG) << "Worspaces size: " << workspaces.size();
  MS_LOG(DEBUG) << "input_workspace_map_ size: " << input_workspace_map_.size();
  std::vector<KernelTensor *> res_workspace(workspaces.begin() + input_workspace_map_.size(), workspaces.end());
  return {res_inputs, res_workspace};
}
}  // namespace mindspore::kernel
