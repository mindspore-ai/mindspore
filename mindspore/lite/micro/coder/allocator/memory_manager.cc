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

#include "coder/allocator/memory_manager.h"
#include <vector>
#include "coder/opcoders/op_coder.h"

namespace mindspore::lite::micro {

static constexpr size_t kDefaultMemAlignSize = 8;

static size_t AlignMemorySize(size_t size) {
  return ((size + kDefaultMemAlignSize - 1) / kDefaultMemAlignSize) * kDefaultMemAlignSize;
}

int MemoryManager::AssignMemory(const std::vector<std::unique_ptr<OperatorCoder>> &nodes) {
  for (const auto &node : nodes) {
    AssignOutputs(node);
    StoreMembufListInfo(node);
    ReleaseInputs(node);
  }
  return RET_OK;
}

void MemoryManager::StoreMembufListInfo(const std::unique_ptr<OperatorCoder> &node) {
  std::vector<MembufPtr> temp;
  for (const auto &membuf : membuf_list_) {
    auto buf = std::make_shared<Membuf>(membuf->key_, membuf->status_, membuf->size_, membuf->offset_);
    temp.emplace_back(buf);
  }
  auto info = std::make_pair(node->node_index(), temp);
  all_membuf_list_info_.emplace_back(info);
}

size_t MemoryManager::GetAllocatedSize() const {
  if (membuf_list_.empty()) {
    return 0;
  }
  return membuf_list_.back()->offset_ + membuf_list_.back()->size_;
}

void MemoryManager::AssignOutputs(const std::unique_ptr<OperatorCoder> &node) {
  for (const auto &output : node->output_tensors()) {
    if (output == nullptr) {
      MS_LOG(ERROR) << "output tensor is nullptr";
      return;
    }
    size_t size = AlignMemorySize(output->Size());
    std::map<size_t, size_t> size_map = GetReusableMembufMap(size);
    if (size_map.empty()) {
      AssignNewMembuf(output, size);
    } else {
      size_t membuf_index = size_map.begin()->second;
      ReuseExistedMembuf(membuf_index, output, size);
    }
  }
}

void MemoryManager::ReleaseInputs(const std::unique_ptr<OperatorCoder> &node) {
  // release node input and workspace
  for (const auto &input : node->input_tensors()) {
    if (input == nullptr) {
      MS_LOG(ERROR) << "input tensor is nullptr";
      return;
    }
    if (input->category() != Tensor::VAR && input->data_c() != nullptr) {
      continue;
    }
    input->DecRefCount();
    if (input->ref_count() > 0) {
      continue;
    }
    auto item = std::find_if(membuf_list_.begin(), membuf_list_.end(),
                             [input](const MembufPtr &membuf) { return membuf->key_ == input; });
    if (item == membuf_list_.end()) {
      continue;
    }
    auto membuf = *item;
    membuf->status_ = kUnused;
  }
  MergeMembuf();
}

void MemoryManager::AssignNewMembuf(Tensor *key, size_t size) {
  MS_LOG(DEBUG) << "assign new membuf: " << size;
  size_t offset = GetAllocatedSize();
  auto membuf = std::make_shared<Membuf>(key, kReused, size, offset);
  membuf_list_.push_back(membuf);
  variables_offset_.insert(std::make_pair(key, offset));
}

void MemoryManager::ReuseExistedMembuf(size_t index, Tensor *key, size_t size) {
  MembufPtr membuf = membuf_list_[index];
  UpdataMembufInfo(membuf, key);
  if (membuf->size_ > size) {
    SplitMembuf(index, size);
  }
}

void MemoryManager::UpdataMembufInfo(const MembufPtr &membuf, Tensor *key) {
  membuf->status_ = kReused;
  membuf->key_ = key;
  variables_offset_.insert(std::make_pair(key, membuf->offset_));
}

void MemoryManager::SplitMembuf(size_t index, size_t size) {
  if (index >= membuf_list_.size()) {
    MS_LOG(ERROR) << "Index out of vector range.";
  }
  auto membuf = membuf_list_[index];
  size_t bias = membuf->size_ - size;
  if (bias < 32) {
    return;  // Res is too smallTensor
  }
  membuf->size_ = size;
  auto new_membuf = std::make_shared<Membuf>(kUnused, bias, membuf->offset_ + membuf->size_);
  (void)membuf_list_.insert(membuf_list_.begin() + index + 1, new_membuf);
}

void MemoryManager::MergeMembuf() {
  if (membuf_list_.empty()) {
    return;
  }
  std::vector<MembufPtr> temp;
  bool is_continue = false;
  for (const auto &membuf : membuf_list_) {
    if (membuf->status_ == kReused) {
      temp.emplace_back(membuf);
      is_continue = false;
    } else {
      if (!is_continue) {
        temp.emplace_back(membuf);
        is_continue = true;
      } else {
        auto back = temp.back();
        back->size_ += membuf->size_;
      }
    }
  }
  membuf_list_ = temp;
}

std::map<size_t, size_t> MemoryManager::GetReusableMembufMap(size_t size) {
  std::map<size_t, size_t> size_map;
  for (size_t i = 0; i < membuf_list_.size(); ++i) {
    auto membuf = membuf_list_[i];
    auto index = i;
    if (membuf->status_ == kUnused && membuf->size_ >= size) {
      (void)size_map.insert(std::make_pair(membuf->size_, index));
    }
  }
  return size_map;
}
}  // namespace mindspore::lite::micro
