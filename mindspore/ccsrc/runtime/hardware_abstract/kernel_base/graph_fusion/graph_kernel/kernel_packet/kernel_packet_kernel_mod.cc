/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/kernel_packet/kernel_packet_kernel_mod.h"
#include <utility>
#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "ir/anf.h"
#include "utils/ms_context.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "utils/convert_utils.h"
#include "include/utils/anfalgo.h"
#include "abstract/abstract_value.h"
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/kernel_packet/kernel_packet_infer_functor.h"
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/kernel_packet/kernel_packet_engine.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore::kernel {
namespace {
constexpr char kMindsporeDumpConfig[] = "MINDSPORE_DUMP_CONFIG";
// Note: Same as the implementation in anf_runtime_algorithm.h.
// Placed here only because mindspore_hardware_abstract.so cannot depend on mindspore_backend_common.so.
bool IsLaunchIgnoredInputAddressIdx(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto kernel_mod = kernel_info->MutableKernelMod();
  MS_EXCEPTION_IF_NULL(kernel_mod);
  // Search for the ignore list if kernelmod has ignore list.
  std::vector<size_t> launch_ignored_input_idx = kernel_mod->GetLaunchIgnoredInputAddressIdx();
  if (!launch_ignored_input_idx.empty()) {
    if (std::find(launch_ignored_input_idx.begin(), launch_ignored_input_idx.end(), input_idx) !=
        launch_ignored_input_idx.end()) {
      return true;
    }
    return false;
  }

  // The new ignore input cannot be dumped, so it is not ignored when dump.
  static bool is_enable_dump = !common::GetEnv(kMindsporeDumpConfig).empty();
  if (is_enable_dump) {
    return false;
  }

  auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, input_idx);
  auto cnode = kernel_with_index.first;
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode->abstract());
  const auto &input_type = cnode->abstract()->BuildType();
  // Tensor or tuple of tensor should not be ignored.
  if (input_type->type_id() == kObjectTypeTensorType) {
    return false;
  }

  if (input_type->type_id() == kObjectTypeTuple) {
    auto type = input_type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(type);
    if (type->dynamic_len()) {
      return false;
    }
    const auto &elements = type->elements();
    if (!elements.empty() && elements[0]->type_id() == kObjectTypeTensorType) {
      return false;
    }
  }

  if (input_type->type_id() == kObjectTypeList) {
    auto type = input_type->cast<ListPtr>();
    MS_EXCEPTION_IF_NULL(type);
    if (type->dynamic_len()) {
      return false;
    }
    const auto &elements = type->elements();
    if (!elements.empty() && elements[0]->type_id() == kObjectTypeTensorType) {
      return false;
    }
  }

  return true;
}
}  // namespace

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
  packet_kernel_mod->inputs_cache_.resize(input_tensor_num, nullptr);
  packet_kernel_mod->is_dynamic_shape_.resize(input_tensor_num, false);
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
      continue;
    } else if (value == nullptr || value->isa<ValueAny>()) {
      infer->SetInnerInput(i, prev_node->abstract());
    } else {
      // const value is moved to parameter, this branch may be dead code.
      MS_LOG(INFO) << "The input " << i << " is a const value: " << value->ToString();
    }
    auto kernel_tensor = std::make_shared<KernelTensor>(abs->GetShape(), abs->GetType(), value);
    packet_kernel_mod->is_dynamic_shape_[i] = abs->GetShape()->IsDynamic();
    packet_kernel_mod->inputs_cache_[i] = std::move(kernel_tensor);
  }
  packet_kernel_mod->real_kernel_mod_ = real_kernel_mod;
  for (size_t input_idx = 0; input_idx < input_tensor_num; ++input_idx) {
    if (IsLaunchIgnoredInputAddressIdx(real_node, input_idx)) {
      packet_kernel_mod->ignored_input_idx_.emplace_back(input_idx);
    }
  }

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
      MS_EXCEPTION_IF_NULL(ori);
      auto shape = is_dynamic_shape_[i] ? host_value_cache_[i]->ToAbstract()->GetShape() : ori->GetShape();
      MS_LOG(DEBUG) << "Inner input " << i << " is host value: " << host_value_cache_[i]->ToString()
                    << ". Its shape is " << shape->ToString() << ", the type is " << ori->GetType();
      std::string device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
      auto device_address = std::make_shared<device::DeviceAddress>(device::GetDeviceTypeByName(device_name));
      inputs_cache_[i] = std::make_shared<KernelTensor>(device_address, nullptr, shape, ori->GetType(), kValueAny);
      if (inputs_cache_[i]->user_data() == nullptr) {
        inputs_cache_[i]->set_user_data(std::make_shared<UserData>());
      }
      inputs_cache_[i]->user_data()->set<std::pair<ValuePtr, bool>>(
        "variable_host_value", std::make_shared<std::pair<ValuePtr, bool>>(host_value_cache_[i], true));
      inner_inputs[i] = inputs_cache_[i].get();
    } else {
      // const value is moved to parameter, this branch may be dead code.
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
  for (size_t i = 0; i < inner_input_num; i++) {
    if (input_map_.count(i) != 0) {
      continue;
    }
    if (inner_inputs[i]->size() > 0 &&
        std::find(ignored_input_idx_.begin(), ignored_input_idx_.end(), i) == ignored_input_idx_.end()) {
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
      MS_LOG(DEBUG) << "Inner input " << i << " used outer input " << j;
      res_inputs[i] = inputs[j];
    } else if (auto iter = input_workspace_map_.find(i); iter != input_workspace_map_.end()) {
      auto j = iter->second;
      MS_LOG(DEBUG) << "Inner input " << i << " used workspace " << j;
      res_inputs[i] = inputs_cache_[i].get();
      // set the device_ptr of workspaces to res_input
      res_inputs[i]->set_pointer_ref_count(workspaces[j]);
      // copy host data to device
      if (host_data_cache_[i] != nullptr) {
        MS_LOG(DEBUG) << "Copy input " << i << " from host to device. device_ptr: " << res_inputs[i]->device_ptr()
                      << ", size: " << res_inputs[i]->size();
        CopyHostToDevice(res_inputs[i]->device_ptr(), host_data_cache_[i], res_inputs[i]->size(), stream_ptr);
      }
    } else {
      // const value is moved to parameter, this branch may be dead code.
      MS_LOG(DEBUG) << "Inner input " << i << " is not found in input_map and input_workspace_map.";
      res_inputs[i] = inputs_cache_[i].get();
    }
  }
  MS_LOG(DEBUG) << "Workspaces size: " << workspaces.size();
  MS_LOG(DEBUG) << "input_workspace_map_ size: " << input_workspace_map_.size();
  std::vector<KernelTensor *> res_workspace(workspaces.begin() + input_workspace_map_.size(), workspaces.end());
  return {res_inputs, res_workspace};
}
CNodePtr GetKernelPacketRealNode(const AnfNodePtr &kernelpacket) {
  auto func_graph = common::AnfAlgo::GetNodeAttr<FuncGraphPtr>(kernelpacket, kAttrFuncGraph);
  MS_EXCEPTION_IF_NULL(func_graph);
  if (func_graph->symbol_engine() == nullptr) {
    // rebuild symbol engine after reload mindir
    auto symbol_engine = mindspore::graphkernel::packet::KernelPacketEngine::Build(func_graph);
    func_graph->set_symbol_engine(symbol_engine);
  }
  auto real_node = func_graph->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(real_node);
  return real_node;
}
}  // namespace mindspore::kernel
