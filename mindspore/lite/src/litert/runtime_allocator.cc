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

#include "src/litert/runtime_allocator.h"

namespace mindspore {
RuntimeAllocator::RuntimeAllocator(size_t aligned_size) {
  aligned_size_ = aligned_size;
  return;
}

RuntimeAllocator::~RuntimeAllocator() {
  if (data_ != nullptr) {
    free(data_);
    data_ = nullptr;
  }
}

void *RuntimeAllocator::MallocOptData() {
  if (data_ == nullptr) {
    data_ = malloc(total_size_);
  }
  return data_;
}

size_t RuntimeAllocator::FindMinFree(size_t size) {
  size_t min_size = total_size_ + 1;
  size_t min_addr = total_size_ + 1;
  for (auto const &itr : free_list_) {
    if (itr.second >= size && min_size > itr.second) {
      min_size = itr.second;
      min_addr = itr.first;
    }
  }
  return min_addr;
}

void RuntimeAllocator::FreeTensorData(lite::Tensor *tensor) {
  size_t offset = offset_map_[tensor];
  free_list_[offset] = used_list_[offset];
  used_list_.erase(offset);

  size_t length = free_list_[offset];

  size_t post_offset = offset + length;
  auto post_iter = free_list_.find(post_offset);
  if (post_iter != free_list_.end()) {
    size_t post_length = post_iter->second;
    free_list_[offset] = length + post_length;
    free_list_.erase(post_offset);
  }

  auto pre_iter = free_list_.lower_bound(offset);
  if (pre_iter != free_list_.begin()) {
    pre_iter--;
    size_t pre_offset = pre_iter->first;
    if ((pre_offset + free_list_[pre_offset]) == offset) {
      free_list_[pre_offset] = free_list_[pre_offset] + length;
      free_list_.erase(offset);
    }
  }
}

void RuntimeAllocator::SetDataOffset(lite::Tensor *tensor, size_t offset) {
  offset_map_[tensor] = offset;
  return;
}

void RuntimeAllocator::Clear(AllocatorPtr default_allocator) {
  total_size_ = 0;
  for (auto iter : offset_map_) {
    iter.first->set_allocator(default_allocator);
    iter.first->set_data(nullptr);
  }
  if (data_ != nullptr) {
    free(data_);
    data_ = nullptr;
  }
  offset_map_.clear();
  free_list_.clear();
  used_list_.clear();
}

void RuntimeAllocator::MallocTensorData(lite::Tensor *tensor) {
  size_t size = tensor->Size();
  size_t offset = FindMinFree(size);

  if (offset > total_size_) {
    if (free_list_.empty()) {
      offset = total_size_;
    } else {
      offset = free_list_.rbegin()->first;
      if (offset + free_list_[offset] < total_size_) {
        offset = total_size_;
      } else {
        free_list_.erase(offset);
      }
    }
    total_size_ = offset + size;
  } else {
    if (free_list_[offset] > size) {
      free_list_[offset + size] = free_list_[offset] - size;
    }
    free_list_.erase(offset);
  }

  used_list_[offset] = size;
  offset_map_[tensor] = offset;
}
}  // namespace mindspore
