/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/micro/coder/allocator/allocator.h"
#include <string>
#include <map>
#include "tools/converter/micro/coder/allocator/memory_manager.h"
#include "tools/converter/micro/coder/opcoders/op_coder.h"

namespace mindspore::lite::micro {
namespace {
const std::map<TypeId, size_t> size_map = {{kNumberTypeFloat, sizeof(float)},   {kNumberTypeFloat32, sizeof(float)},
                                           {kNumberTypeInt32, sizeof(int32_t)}, {kNumberTypeInt16, sizeof(int16_t)},
                                           {kNumberTypeInt8, sizeof(int8_t)},   {kNumberTypeUInt8, sizeof(uint8_t)}};
}
void *MemoryAllocator::MallocWeightTensor(TypeId type_id, size_t size, MallocType type,
                                          const std::string &tensor_name) {
  auto item = size_map.find(type_id);
  MS_CHECK_TRUE_RET_NULL(item != size_map.end(), "unsupported type index");

  size_t type_size = item->second;
  MS_CHECK_TRUE_RET_NULL(type_size > 0, "type size should be greater than 0");
  std::vector<int> shape = {1, static_cast<int>(size / type_size)};
  auto cate = type == kOfflinePackWeight ? lite::Category::CONST_TENSOR : lite::Category::VAR;
  Tensor *weight = new (std::nothrow) lite::Tensor(type_id, shape, mindspore::NHWC, cate);
  MS_CHECK_PTR_RET_NULL(weight);
  weight->set_tensor_name(tensor_name);
  std::string runtime_addr = kWeightPrefixName + std::to_string(weight_index_++);
  malloc_weights_addr_.insert(std::make_pair(weight, runtime_addr));
  if (type == kOfflinePackWeight) {
    saved_weights_addr_.insert(std::make_pair(runtime_addr, weight));
  }
  MS_CHECK_RET_CODE_RET_NULL(weight->MallocData(), "weight malloc data failed!");
  return weight->data();
}

Tensor *MemoryAllocator::MallocTensor(TypeId data_type, const std::vector<int> &shape) {
  auto result = new Tensor(data_type, shape);
  size_t size = result->ElementsNum() * size_map.at(data_type);
  AssignWorkspaces(result, size);
  std::string addr = workspaces_addr_.at(result);
  malloc_weights_addr_.insert(std::make_pair(result, addr));
  return result;
}

void MemoryAllocator::FreeTensor(Tensor *t) {
  if (t == nullptr) return;
  std::string addr = GetRuntimeAddr(t);
  saved_weights_addr_.erase(addr);
  origin_weights_addr_.erase(t);
  bool t_origin = (malloc_weights_addr_.find(t) != malloc_weights_addr_.end());
  malloc_weights_addr_.erase(t);
  if (t_origin) {
    delete t;
  }
}

void MemoryAllocator::Free() {
  for (auto iter = malloc_weights_addr_.begin(); iter != malloc_weights_addr_.end();) {
    Tensor *tensor = iter->first;
    if (origin_weights_addr_.find(tensor) == origin_weights_addr_.end()) {
      delete tensor;
      malloc_weights_addr_.erase(iter++);
    } else {
      iter++;
    }
  }
  for (auto &item : allocated_) {
    free(item);
    item = nullptr;
  }
  workspaces_addr_.clear();
  workspace_size_ = 0;
  tensors_size_ = 0;
  weight_index_ = 0;
  is_next_ = false;
  offset_ = 0;
  allocated_.clear();
  saved_weights_addr_.clear();
  origin_weights_addr_.clear();
  malloc_weights_addr_.clear();
  tensors_addr_.clear();
}

std::map<Tensor *, std::string> MemoryAllocator::tensors_map() const {
  std::map<Tensor *, std::string> res;
  res.insert(tensors_addr_.begin(), tensors_addr_.end());
  res.insert(malloc_weights_addr_.begin(), malloc_weights_addr_.end());
  if (Configurator::GetInstance()->code_mode() == CodeMode::Train) {
    // in order to put all weights into struct ModelParameter model_params
    for (const auto &iter : saved_weights_addr_) {
      res.insert({iter.second, iter.first});
    }
  }
  return res;
}

void MemoryAllocator::AssignWorkspaces(void *addr, size_t size) {
  if (is_next_) {
    is_next_ = false;
    offset_ = 0;
  }
  workspaces_addr_.insert(std::make_pair(addr, kBufferPrefixNameAdd + std::to_string(tensors_size_ + offset_)));
  offset_ += size;
  if (workspace_size_ < offset_) {
    workspace_size_ = offset_;
  }
}

void MemoryAllocator::RecordTensorsAddr(const std::map<Tensor *, size_t> &offsets) {
  for (auto &item : offsets) {
    auto tensor = item.first;
    auto offset = item.second;
    tensors_addr_.insert(std::make_pair(tensor, kBufferPrefixNameAdd + std::to_string(offset)));
  }
}

void MemoryAllocator::AssignGraphInputs(const std::vector<Tensor *> &inputs) {
  size_t num = inputs.size();
  for (size_t i = 0; i < num; ++i) {
    tensors_addr_.insert(std::make_pair(inputs.at(i), kInputPrefixName + std::to_string(i)));
  }
}

void MemoryAllocator::RecordOriginWeightsAddr(const std::vector<std::unique_ptr<OperatorCoder>> &nodes) {
  for (const auto &node : nodes) {
    std::vector<Tensor *> inputs = node->input_tensors();
    for (const auto &tensor : inputs) {
      if (tensor->category() == lite::Category::CONST_TENSOR || tensor->category() == lite::Category::CONST_SCALAR) {
        std::string runtime_addr = kWeightPrefixName + std::to_string(weight_index_);
        origin_weights_addr_.insert(std::make_pair(tensor, runtime_addr));
        weight_index_++;
      }
    }
  }
}

int MemoryAllocator::AssignTensors(const std::vector<std::unique_ptr<OperatorCoder>> &nodes) {
  // intend to support multi memory assign algorithm
  auto manager = std::make_unique<MemoryManager>();
  int ret = manager->AssignMemory(nodes);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "assign memory failed";
    return RET_ERROR;
  }
  std::map<Tensor *, size_t> offsets = manager->variables_offset();
  RecordTensorsAddr(offsets);

  tensors_size_ = manager->GetAllocatedSize();
  return RET_OK;
}

int MemoryAllocator::Assign(const std::vector<Tensor *> &inputs,
                            const std::vector<std::unique_ptr<OperatorCoder>> &nodes) {
  AssignGraphInputs(inputs);
  RecordOriginWeightsAddr(nodes);
  return AssignTensors(nodes);
}
}  // namespace mindspore::lite::micro
