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

namespace mindspore::kernel {
constexpr size_t kShapeTypeSize = sizeof(int64_t);

bool kernelpacket::Init(KernelPacketInner *kernel_packet, const CNodePtr &real_node) {
  MS_EXCEPTION_IF_NULL(real_node);
  kernel_packet->real_node_name_ = real_node->DebugString();
  FuncGraphPtr func_graph = real_node->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Empty func_graph of " << kernel_packet->real_node_name_;
    return false;
  }
  auto symbol_engine = func_graph->symbol_engine();
  if (symbol_engine == nullptr) {
    MS_LOG(ERROR) << "Empty symbol engine of func_graph of " << kernel_packet->real_node_name_;
    return false;
  }
  size_t input_tensor_num = common::AnfAlgo::GetInputTensorNum(real_node);
  kernel_packet->inputs_cache_.reserve(input_tensor_num);
  for (size_t i = 0; i < input_tensor_num; i++) {
    auto abs = real_node->input(i + 1)->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto kernel_tensor = std::make_shared<KernelTensor>(abs->GetShape(), abs->GetType(), abs->GetValue());
    (void)kernel_packet->inputs_cache_.emplace_back(kernel_tensor);
  }
  kernel_packet->input_node_map_.clear();
  auto outer_inputs = func_graph->parameters();
  // initialize input index and workspace index
  for (size_t i = 0; i < input_tensor_num; ++i) {
    auto prev_node = real_node->input(i + 1);
    MS_LOG(DEBUG) << "The realnode " << real_node->DebugString() << " input[" << i << "] is "
                  << prev_node->DebugString();
    auto iter = std::find(outer_inputs.begin(), outer_inputs.end(), prev_node);
    if (iter != outer_inputs.end()) {
      kernel_packet->input_map_[i] = static_cast<size_t>(iter - outer_inputs.begin());
    } else {
      // Skip value node
      if (!symbol_engine->IsDependValue(prev_node)) {
        auto value_node = prev_node->cast<ValueNodePtr>();
        if (value_node == nullptr) {
          MS_LOG(ERROR) << "The input[" << i << "] of " << real_node->DebugString()
                        << " is not one of [outer input, depend on value, value node]";
          return false;
        }
        if (value_node->value()->isa<ValueAny>()) {
          MS_LOG(ERROR) << "ValueAny in " << i << "th input of " << real_node->DebugString();
          return false;
        }
        MS_LOG(DEBUG) << "The realnode's input[" << i
                      << "] is not value-depend. value_node: " << value_node->value()->ToString();
        continue;
      }
      MS_EXCEPTION_IF_NULL(prev_node->abstract());
      kernel_packet->input_node_map_[i] = SimpleNode{prev_node->abstract(), prev_node->DebugString()};
    }
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(real_node->kernel_info());
  if (kernel_info == nullptr) {
    MS_LOG(ERROR) << "The realnode " << real_node->DebugString() << " has no kernel info";
    return false;
  }
  kernel_packet->real_kernel_mod_ = kernel_info->GetKernelMod();
  return true;
}

/// \brief convert int value or int value list to ShapeVector
/// \return if success
bool ValueToShape(const ValuePtr &value, ShapeVector *shape) {
  if (value->isa<Int32Imm>() || value->isa<Int64Imm>()) {
    auto v = AnfUtils::GetIntValue(value);
    shape->push_back({v});
  } else if (value->isa<ValueSequence>()) {
    auto vec = value->cast<ValueSequencePtr>()->value();
    if (vec.empty()) {
      return true;
    } else if (vec[0]->isa<Int32Imm>() || vec[0]->isa<Int64Imm>()) {
      shape->reserve(vec.size());
      for (auto v : vec) {
        auto v1 = AnfUtils::GetIntValue(v);
        shape->push_back(v1);
      }
    } else {
      return false;
    }
  } else if (value->isa<tensor::Tensor>()) {
    *shape = CheckAndConvertUtils::CheckTensorIntValue("value", value, "KernelPacket");
  } else {
    return false;
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

std::pair<int, bool> KernelPacketKernelMod::QuerySymbolicValue(size_t i, const AbstractBasePtr &abs) {
  ShapeVector shape;
  auto value_ptr = symshape::QueryValue(abs);
  if (value_ptr == nullptr || value_ptr == kValueAny) {
    MS_LOG(ERROR) << "Symbol engine query value failed";
    return std::make_pair(KRET_RESIZE_FAILED, false);
  }
  MS_LOG(DEBUG) << "Value of input[" << i << "]: " << value_ptr->DumpText();
  host_value_cache_[i] = value_ptr;
  if (!ValueToShape(value_ptr, &shape)) {
    if (value_ptr->isa<BoolImm>()) {
      AllocWorkspace(i, sizeof(bool));
      host_data_cache_[i].push_back(static_cast<int8_t>(GetValue<bool>(value_ptr)));
      MS_LOG(DEBUG) << "Cached the bool value " << value_ptr->ToString();
    } else {
      MS_LOG(DEBUG) << "The value is not bool nor int, skip it: " << value_ptr->ToString();
    }
    return std::make_pair(KRET_OK, false);
  }
  if (shape.empty()) {
    MS_LOG(DEBUG) << "The value of " << i << "th input of inner kernel is empty.";
    return std::make_pair(KRET_OK, true);
  }
  host_data_cache_[i].resize(shape.size() * sizeof(shape[0]));
  (void)memcpy_s(host_data_cache_[i].data(), host_data_cache_[i].size(), shape.data(), shape.size() * sizeof(shape[0]));
  return std::make_pair(KRET_OK, true);
}

int KernelPacketKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  MS_LOG(DEBUG) << "=========================Start to resize: " << kernel_name_ << "=========================";
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_workspace_map_.clear();
  auto inner_input_num = inputs_cache_.size();
  std::vector<KernelTensor *> inner_inputs(inner_input_num, nullptr);
  host_data_cache_.clear();
  host_data_cache_.resize(inner_input_num);
  host_value_cache_.clear();
  host_value_cache_.resize(inner_input_num, nullptr);
  for (size_t i = 0; i < inner_input_num; ++i) {
    if (auto iter = input_map_.find(i); iter != input_map_.end()) {
      MS_LOG(DEBUG) << "Inner input " << i << " -> outer input " << iter->second;
      inner_inputs[i] = inputs[iter->second];
    } else if (auto iter = input_node_map_.find(i); iter != input_node_map_.end()) {
      MS_LOG(DEBUG) << "Inner input " << i << " -> " << iter->second.debug_info;
      auto ori_tensor = inputs_cache_[i];
      inputs_cache_[i] = std::make_shared<KernelTensor>(ori_tensor->GetShape(), ori_tensor->GetType(), nullptr);
      inner_inputs[i] = inputs_cache_[i].get();
      auto query_ret = QuerySymbolicValue(i, iter->second.abs);
      if (query_ret.first != KRET_OK) {
        return query_ret.first;
      }
      auto value_ptr = host_value_cache_[i];
      if (!query_ret.second) {
        inner_inputs[i]->SetValue(value_ptr);
        continue;
      }

      // the data type is int64.
      auto type_id = inner_inputs[i]->type_id();
      size_t data_num = host_data_cache_[i].size() / kShapeTypeSize;
      if (type_id == kObjectTypeTensorType) {
        inner_inputs[i]->SetShapeVector(ShapeVector{static_cast<int64_t>(data_num)});
        MS_LOG(DEBUG) << "The inner_inputs[" << i << "]'s data_num is " << data_num << ", shape is "
                      << inner_inputs[i]->GetShapeVector();
      } else if (type_id == kObjectTypeTuple || type_id == kObjectTypeList) {
        // Case when value is tuple of int
        abstract::BaseShapePtrList shapes(data_num, std::make_shared<abstract::NoShape>());
        auto tuple_shape = std::make_shared<abstract::TupleShape>(std::move(shapes));
        inner_inputs[i]->SetShape(tuple_shape);
        MS_LOG(DEBUG) << "The inner_inputs[" << i << "]'s data_num is " << data_num << ", shape is "
                      << inner_inputs[i]->GetShapeVector();
      } else {
        MS_LOG(DEBUG) << "The inner_inputs[" << i << "]'s type_id is " << type_id;
      }
      // SetHostData, in case some operations use shape data when resize.
      size_t data_size = host_data_cache_[i].size();
      inner_inputs[i]->SetValue(value_ptr);
      AllocWorkspace(i, data_size);
    } else {
      MS_LOG(DEBUG) << "Inner input " << i << " is not found in input_map and input_workspace_map.";
      inner_inputs[i] = inputs_cache_[i].get();
    }
  }

  auto res = real_kernel_mod_->Resize(inner_inputs, outputs);
  MS_LOG(DEBUG) << "Inner kernel resize finished: " << real_node_name_;
  if (res != KRET_OK) {
    return res;
  }
  const auto &workspace = real_kernel_mod_->GetWorkspaceSizeList();
  MS_LOG(DEBUG) << "Inner kernel workspaces size: " << workspace.size();
  workspace_size_list_.reserve(workspace.size() + inner_input_num);
  // Inner kernel's workspace is behind shape workspace
  (void)workspace_size_list_.insert(workspace_size_list_.end(), workspace.begin(), workspace.end());
  MS_LOG(DEBUG) << "=========================finish resize: " << kernel_name_ << "=========================";
  return KRET_OK;
}

bool KernelPacketKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspaces,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(DEBUG) << "=========================start to launch: " << kernel_name_ << "=========================";
  auto [inner_inputs, inner_workspaces] = GetLaunchArgs(inputs, workspaces, stream_ptr);
  auto res = real_kernel_mod_->Launch(inner_inputs, inner_workspaces, outputs, stream_ptr);
  MS_LOG(DEBUG) << "Finish inner kernel launch: " << real_node_name_;
  if (!res) {
    MS_LOG(ERROR) << "Launch kernel: " << real_node_name_ << " failed.";
    return false;
  }
  MS_LOG(DEBUG) << "=========================finish launch: " << kernel_name_ << "=========================";
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
    if (input_map_.count(i) > 0) {
      auto j = input_map_[i];
      MS_LOG(DEBUG) << "Inner input " << i << " -> outer input " << j;
      res_inputs[i] = inputs[j];
    } else if (input_workspace_map_.count(i) > 0) {
      auto j = input_workspace_map_[i];
      MS_LOG(DEBUG) << "Inner input " << i << " -> workspace " << j;
      res_inputs[i] = inputs_cache_[i].get();
      // set the device_ptr of workspaces to res_input
      res_inputs[i]->set_pointer_ref_count(workspaces[j]->pointer_ref_count());
      // copy host data to device
      if (!host_data_cache_[i].empty()) {
        memcpy_async_(res_inputs[i]->device_ptr(), host_data_cache_[i].data(), host_data_cache_[i].size(), stream_ptr);
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
