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

namespace mindspore::kernel {
constexpr size_t kShapeTypeSize = sizeof(int64_t);

/// \brief convert int value or int array value to shape array
/// \return if success
static bool ValueToShape(const ValuePtr &value, ShapeArray *shape) {
  if (value->isa<Int32Imm>() || value->isa<Int64Imm>()) {
    auto v = AnfUtils::GetIntValue(value);
    shape->push_back({v});
  } else if (value->isa<ValueSequence>()) {
    auto vec = value->cast<ValueSequencePtr>()->value();
    if (vec.empty()) {
      shape->emplace_back();
    } else if (vec[0]->isa<Int32Imm>() || vec[0]->isa<Int64Imm>()) {
      ShapeVector shape_vec;
      shape_vec.reserve(vec.size());
      for (auto v : vec) {
        auto v1 = AnfUtils::GetIntValue(v);
        shape_vec.push_back(v1);
      }
      shape->push_back(std::move(shape_vec));
    } else if (vec[0]->isa<ValueSequence>()) {
      for (auto &sub_value : vec) {
        ValueToShape(sub_value, shape);
      }
    } else {
      return false;
    }
  } else if (value->isa<tensor::Tensor>()) {
    auto raw_value_vec = CheckAndConvertUtils::CheckTensorIntValue("value", value, "KernelPacket");
    shape->push_back(std::move(raw_value_vec));
  } else {
    return false;
  }
  return true;
}

int KernelPacketKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  MS_LOG(DEBUG) << "=========================Start to resize: " << kernel_name_ << "=========================";
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto inner_input_num = inputs_cache_.size();
  std::vector<KernelTensor *> inner_inputs(inner_input_num, nullptr);
  // Hold pointer to temp host data, to free them after kernel resize
  std::vector<std::unique_ptr<int64_t[]>> temp_host_datas;
  for (size_t i = 0; i < inner_input_num; ++i) {
    if (auto iter = input_map_.find(i); iter != input_map_.end()) {
      MS_LOG(DEBUG) << "Inner input " << i << " -> "
                    << "outer input " << iter->second;
      inner_inputs[i] = inputs[iter->second];
    } else if (auto iter = input_shape_map_.find(i); iter != input_shape_map_.end()) {
      // Put shape into Input Kerneltensor
      MS_LOG(DEBUG) << "Inner input " << i << "-> " << iter->second.second
                    << "th cnode: " << iter->second.first->DebugString();
      inner_inputs[i] = inputs_cache_[i].get();

      ShapeArray shape_values;
      auto value = symbol_engine_->QueryValue(iter->second.first);
      if (value == nullptr || value == kValueAny) {
        MS_LOG(ERROR) << "Symbol engine query value failed, node: " << iter->second.first->fullname_with_scope();
        return KRET_RESIZE_FAILED;
      }
      MS_LOG(DEBUG) << "Result of QueryValue: " << value->DumpText();
      if (!ValueToShape(value, &shape_values)) {
        inner_inputs[i]->SetValue(value);
        continue;
      }

      auto idx_in_output = iter->second.second;
      if (idx_in_output >= shape_values.size()) {
        MS_LOG(ERROR) << "The " << i << "th input of inner kernel are " << idx_in_output
                      << "th of output of prev node, but that output's size is only " << shape_values.size();
        return KRET_RESIZE_FAILED;
      }
      ShapeVector shape_vector = {SizeToLong(shape_values[idx_in_output].size())};
      auto type_id = inner_inputs[i]->type_id();
      if (type_id == kObjectTypeTensorType) {
        inner_inputs[i]->SetShapeVector(shape_vector);
      }
      // SetHostData, in case some operations use shape data when resize.
      size_t data_size = kShapeTypeSize * shape_vector[0];
      auto data_p = std::make_unique<int64_t[]>(shape_vector[0]);
      int64_t *raw_p = data_p.get();
      (void)memcpy(raw_p, shape_values[idx_in_output].data(), data_size);
      temp_host_datas.push_back(std::move(data_p));
      auto shape_address = std::make_shared<Address>(raw_p, data_size);
      inner_inputs[i]->SetHostData(shape_address);
      inner_inputs[i]->SetValue(MakeValue(shape_values[idx_in_output]));

      // Alloc intermediate shape space
      auto input_idx = iter->first;
      shape_cache_[input_idx] = shape_values[idx_in_output];
      MS_LOG(DEBUG) << "Shape size: " << data_size;
      input_workspace_map_[input_idx] = workspace_size_list_.size();
      workspace_size_list_.push_back(data_size);
    } else {
      MS_LOG(ERROR) << "The " << i
                    << "th input of inner kernel is neither from outer input nor from shape of other inner nodes ";
      return KRET_RESIZE_FAILED;
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
  // Put shapes to workspace
  for (auto [input_idx, workspace_idx] : input_workspace_map_) {
    if (workspaces.size() <= workspace_idx) {
      MS_LOG(ERROR) << "Worksapce_idx " << workspace_idx << " is greater than worksapces size " << workspaces.size();
      return false;
    }
    auto res = memcpy_async_(workspaces[workspace_idx]->device_ptr(), shape_cache_[input_idx].data(),
                             workspaces[workspace_idx]->size(), stream_ptr);
    if (!res) {
      MS_LOG(ERROR) << "Launch " << real_node_name_ << "failed!";
      return false;
    }
  }
  MS_LOG(DEBUG) << "Memcpy finished";

  auto [inner_inputs, inner_workspaces, inner_outputs] = GetLaunchArgs(inputs, workspaces, outputs);
  auto res = real_kernel_mod_->Launch(inner_inputs, inner_workspaces, inner_outputs, stream_ptr);
  MS_LOG(DEBUG) << "Finish inner kernel launch: " << real_node_name_;
  if (!res) {
    MS_LOG(ERROR) << "Launch kernel: " << real_node_name_ << " failed!!!!!";
    return false;
  }
  MS_LOG(DEBUG) << "=========================finish launch: " << kernel_name_ << "=========================";
  return true;
}

std::vector<KernelAttr> KernelPacketKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
  return support_list;
}

bool KernelPacketKernelMod::Init(const CNodePtr &real_node) {
  real_node_name_ = real_node->fullname_with_scope();
  FuncGraphPtr func_graph = real_node->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Empty func_graph of " << real_node_name_;
    return false;
  }
  symbol_engine_ = func_graph->symbol_engine();
  if (symbol_engine_ == nullptr) {
    MS_LOG(ERROR) << "Empty symbol engine of func_graph of " << real_node_name_;
    return false;
  }
  auto attr = primitive_->GetAttr(kAttrKernelPacketNode);
  if (attr == nullptr) {
    MS_LOG(ERROR) << "KernelPacketNode has no attr kernel_packet";
    return false;
  }
  kernel_name_ = GetValue<std::string>(attr);
  MS_LOG(DEBUG) << "KernelPacketKernelMod init: " << kernel_name_;
  auto kernel_inputs = AnfAlgo::GetOrCreateAllInputKernelTensors(real_node);
  inputs_cache_.reserve(kernel_inputs.size());
  for (auto kernel_input : kernel_inputs) {
    auto input = std::make_shared<KernelTensor>(*kernel_input);
    inputs_cache_.push_back(std::move(input));
  }
  auto outer_inputs = func_graph->parameters();
  // initialize input index and workspace index
  for (size_t i = 0; i < common::AnfAlgo::GetInputTensorNum(real_node); ++i) {
    MS_LOG(DEBUG) << "Input " << i << " :";
    auto [prev_node, prev_out_idx] = common::AnfAlgo::GetPrevNodeOutput(real_node, i);
    MS_LOG(DEBUG) << prev_out_idx << "th output of " << prev_node->DebugString();
    auto iter = std::find(outer_inputs.begin(), outer_inputs.end(), prev_node);
    if (iter != outer_inputs.end()) {
      input_map_[i] = iter - outer_inputs.begin();
    } else {
      // Skip value node
      if (!symbol_engine_->IsDependValue(prev_node)) {
        auto value_node = prev_node->cast<ValueNodePtr>();
        if (value_node == nullptr) {
          MS_LOG(ERROR) << "The " << i << "th input of " << real_node_name_
                        << " is not one of [outer input, depend on value, value node]";
          return false;
        }
        if (value_node->value()->isa<ValueAny>()) {
          MS_LOG(ERROR) << "Value any in " << i << "th input of " << real_node_name_;
          return false;
        }
        continue;
      }
      input_shape_map_[i] = {prev_node, prev_out_idx};
    }
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(real_node->kernel_info());
  if (kernel_info == nullptr) {
    MS_LOG_ERROR << "Real node: " << real_node_name_ << " has no kernel info";
  }
  real_kernel_mod_ = kernel_info->GetKernelMod();
  return true;
}

KernelPacketKernelMod::AddressArgs KernelPacketKernelMod::GetLaunchArgs(const std::vector<KernelTensor *> &inputs,
                                                                        const std::vector<KernelTensor *> &workspaces,
                                                                        const std::vector<KernelTensor *> &outputs) {
  std::vector<KernelTensor *> res_inputs;
  res_inputs.resize(inputs_cache_.size(), nullptr);
  for (auto [i, j] : input_map_) {
    MS_LOG(DEBUG) << "Inner input -> outer input: " << i << " -> " << j;
    res_inputs[i] = inputs[j];
  }
  for (auto [i, j] : input_workspace_map_) {
    MS_LOG(DEBUG) << "Inner input -> workspace: " << i << " -> " << j;
    res_inputs[i] = workspaces[j];
  }

  std::vector<KernelTensor *> res_workspace;
  MS_LOG(DEBUG) << "Worspaces size: " << workspaces.size();
  MS_LOG(DEBUG) << "shape_cache size: " << shape_cache_.size();
  (void)res_workspace.insert(res_workspace.end(), workspaces.begin() + shape_cache_.size(), workspaces.end());

  return {res_inputs, res_workspace, outputs};
}
}  // namespace mindspore::kernel
