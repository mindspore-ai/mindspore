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
const std::map<TypeId, size_t> size_map = {{kNumberTypeFloat, sizeof(float)},     {kNumberTypeFloat32, sizeof(float)},
                                           {kNumberTypeInt32, sizeof(int32_t)},   {kNumberTypeInt16, sizeof(int16_t)},
                                           {kNumberTypeInt8, sizeof(int8_t)},     {kNumberTypeUInt8, sizeof(uint8_t)},
                                           {kNumberTypeFloat16, sizeof(uint16_t)}};
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
  for (auto &item : auxiliary_weights_) {
    delete item.second.first;
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
  origin_weights_.clear();
  auxiliary_weights_.clear();
}

std::map<Tensor *, std::string> MemoryAllocator::tensors_map() const {
  std::map<Tensor *, std::string> res;
  res.insert(tensors_addr_.begin(), tensors_addr_.end());
  res.insert(malloc_weights_addr_.begin(), malloc_weights_addr_.end());
  (void)std::for_each(
    auxiliary_weights_.begin(), auxiliary_weights_.end(),
    [&res](const std::pair<Tensor *, std::pair<Tensor *, std::string>> &item) { res.insert(item.second); });
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

int MemoryAllocator::RecordOriginWeightsAddr(const std::vector<Tensor *> &all_tensors,
                                             const std::string &changeable_weights_name) {
  std::vector<std::string> weights_name;
  if (!changeable_weights_name.empty()) {
    weights_name = StrSplit(changeable_weights_name, ",");
  }
  for (const auto &tensor : all_tensors) {
    if (tensor->category() == lite::Category::CONST_TENSOR || tensor->category() == lite::Category::CONST_SCALAR) {
      if (std::find(weights_name.begin(), weights_name.end(), tensor->tensor_name()) != weights_name.end()) {
        if (RecordChangeableWeights(tensor) != RET_OK) {
          MS_LOG(ERROR) << "RecordChangeableWeights for " << tensor->tensor_name() << " failed.";
          return RET_ERROR;
        }
      }
      std::string runtime_addr = kWeightPrefixName + std::to_string(weight_index_++);
      origin_weights_addr_.insert(std::make_pair(tensor, runtime_addr));
      origin_weights_.push_back(tensor);
    }
  }
  return RET_OK;
}

int MemoryAllocator::AssignTensors(const std::vector<std::unique_ptr<OperatorCoder>> &nodes,
                                   const std::vector<Tensor *> &outputs) {
  // intend to support multi memory assign algorithm
  auto manager = std::make_unique<MemoryManager>();
  int ret = manager->AssignMemory(nodes, outputs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "assign memory failed";
    return RET_ERROR;
  }
  std::map<Tensor *, size_t> offsets = manager->variables_offset();
  RecordTensorsAddr(offsets);

  tensors_size_ = manager->GetAllocatedSize();
  return RET_OK;
}

int MemoryAllocator::Assign(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                            const std::vector<std::unique_ptr<OperatorCoder>> &nodes,
                            const std::vector<Tensor *> &all_tensors, const std::string &changeable_weights_name) {
  AssignGraphInputs(inputs);
  if (RecordOriginWeightsAddr(all_tensors, changeable_weights_name) != RET_OK) {
    MS_LOG(ERROR) << "RecordOriginWeightsAddr failed.";
    return RET_ERROR;
  }
  return AssignTensors(nodes, outputs);
}

void MemoryAllocator::MarkSharedWeight(const Tensor *src, void *pack_weight) {
  shared_pack_weights_[src] = pack_weight;
}

void *MemoryAllocator::GetSharedWeightAddr(const Tensor *src) {
  return shared_pack_weights_.find(src) == shared_pack_weights_.end() ? nullptr : shared_pack_weights_[src];
}

int MemoryAllocator::RecordChangeableWeights(Tensor *src) {
  MS_ASSERT(src != nullptr);
  auto variable_str = GetAuxiliaryWeight(src);
  if (!variable_str.empty()) {
    return RET_OK;
  }
  if (!src->IsConst()) {
    MS_LOG(ERROR) << "Currently, the tensor must be a constant.";
    return RET_NOT_SUPPORT;
  }
  auto shape = src->shape();
  auto shape_tensor = new (std::nothrow)
    Tensor(kNumberTypeInt32, {static_cast<int>(shape.size())}, src->format(), Category::CONST_TENSOR);
  if (shape_tensor == nullptr) {
    MS_LOG(ERROR) << "Create an assistant tensor failed.";
    return RET_NULL_PTR;
  }
  auto data = shape_tensor->MutableData();
  if (data == nullptr) {
    MS_LOG(ERROR) << "Create an assistant tensor failed.";
    delete shape_tensor;
    return RET_NULL_PTR;
  }
  if (memcpy_s(data, shape_tensor->Size(), shape.data(), shape.size() * sizeof(int)) != EOK) {
    MS_LOG(ERROR) << "Create an assistant tensor failed.";
    delete shape_tensor;
    return RET_ERROR;
  }
  shape_tensor->set_tensor_name(src->tensor_name() + "_shape");
  std::string runtime_addr = kWeightPrefixName + std::to_string(weight_index_++);
  auxiliary_weights_[src] = std::make_pair(shape_tensor, runtime_addr);
  return RET_OK;
}

std::string MemoryAllocator::GetAuxiliaryWeight(Tensor *src) {
  auto iter = auxiliary_weights_.find(src);
  if (iter != auxiliary_weights_.end()) {
    return iter->second.second;
  }
  return {};
}
}  // namespace mindspore::lite::micro
