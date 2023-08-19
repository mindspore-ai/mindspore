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

#include "coder/dynamic_mem_manager.h"
#include <vector>
#include <utility>
#include "coder/allocator/memory_manager.h"
#include "coder/generator/component/component.h"

namespace mindspore::lite::micro {
int DynamicMemManager::AllocDynamicMem(const std::vector<std::unique_ptr<OperatorCoder>> &nodes,
                                       const std::vector<Tensor *> &graph_inputs,
                                       const std::vector<Tensor *> &graph_outputs,
                                       const ShapeInfoContainer *shape_info_container) {
  MS_CHECK_TRUE_MSG(shape_info_container, RET_NULL_PTR, "ShapeInfoContainer is a nullptr.");
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    MS_CHECK_TRUE_MSG(kInputPrefixName != nullptr, RET_NULL_PTR, "Input is a nullptr.");
    graph_inputs_.insert(std::make_pair(graph_inputs.at(i), kInputPrefixName + std::to_string(i)));
  }
  auto var_tensor_shapes = shape_info_container->GetVarTensorInfos();
  MS_CHECK_TRUE_MSG(!var_tensor_shapes.empty(), RET_ERROR, "Cannot get var-tensor's shape-info");
  auto scene_num = var_tensor_shapes.begin()->second.size();
  for (const auto &item : var_tensor_shapes) {
    MS_CHECK_TRUE_MSG(item.first, RET_NULL_PTR, "Find a nullptr in shape-infos");
    MS_CHECK_TRUE_MSG(item.second.size() == scene_num, RET_ERROR, "Shape-info is invalid.");
  }
  for (size_t i = 0; i < scene_num; ++i) {
    for (const auto &item : var_tensor_shapes) {
      const_cast<Tensor *>(item.first)->ResetRefCount();
      const_cast<Tensor *>(item.first)->set_shape(item.second[i]);
    }
    auto ret = AllocDynamicMemCore(nodes, graph_outputs, i);
    MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "Alloc dynamic memory failed.");
  }
  return RET_OK;
}

int DynamicMemManager::AllocDynamicMemCore(const std::vector<std::unique_ptr<OperatorCoder>> &nodes,
                                           const std::vector<Tensor *> &graph_outputs, int scene_index) {
  if (offsets_all_scenes_.find(scene_index) != offsets_all_scenes_.end()) {
    MS_LOG(ERROR) << "Current scene has been processed.";
    return RET_ERROR;
  }
  auto manager = std::make_unique<MemoryManager>();
  int ret = manager->AssignMemory(nodes, graph_outputs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "assign memory failed";
    return RET_ERROR;
  }
  std::map<Tensor *, size_t> offsets = manager->variables_offset();
  if (offset_index_.empty()) {
    int index = 0;
    for (auto &item : offsets) {
      offset_index_[item.first] = index++;
      offsets_all_scenes_[scene_index].push_back(item.second);
    }
  } else {
    MS_CHECK_TRUE_MSG(offsets.size() == offset_index_.size(), RET_ERROR, "Tensors num is not same.");
    for (auto &item : offsets) {
      MS_CHECK_TRUE_MSG(offset_index_.find(item.first) != offset_index_.end(), RET_ERROR, "Tensor cannot be found.");
      offsets_all_scenes_[scene_index].push_back(item.second);
    }
  }
  buffer_sizes_.push_back(manager->GetAllocatedSize());
  offsets_all_scenes_[scene_index].emplace_back(manager->GetAllocatedSize());
  return RET_OK;
}

std::string DynamicMemManager::GetVarTensorAddr(const Tensor *tensor) const {
  if (graph_inputs_.find(tensor) != graph_inputs_.end()) {
    return graph_inputs_.at(tensor);
  }
  if (offset_index_.find(tensor) == offset_index_.end()) {
    return "";
  }
  if (kBufferPrefixName == nullptr || kOffsetPrefixName == nullptr) {
    MS_LOG(ERROR) << "Buffer or Offset is a nullptr.";
    return "";
  }
  return std::string(kBufferPrefixName) + " + " + kOffsetPrefixName + "[" + std::to_string(offset_index_.at(tensor)) +
         "]";
}

std::string DynamicMemManager::AllocWorkSpace(size_t size, int index) {
  if (index < 0 || static_cast<size_t>(index) >= buffer_sizes_.size()) {
    return "";
  }
  if (static_cast<size_t>(index) + 1 >= workspaces_.size()) {
    workspaces_.insert(workspaces_.end(), index + 1 - workspaces_.size(), 0);
  }
  if (workspaces_[index] < size) {
    workspaces_[index] = size;
  }
  if (kBufferPrefixName == nullptr) {
    MS_LOG(ERROR) << "Buffer is a nullptr.";
    return "";
  }
  if (kOffsetPrefixName == nullptr) {
    MS_LOG(ERROR) << "Offset is a nullptr.";
    return "";
  }
  return "(" + std::string(kBufferPrefixName) + " + " + std::string(kBufferPrefixName) + "[" +
         std::to_string(offsets_all_scenes_.begin()->second.size() - 1) + "])";
}
}  // namespace mindspore::lite::micro
