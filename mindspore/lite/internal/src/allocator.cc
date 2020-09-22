/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "internal/src/allocator.h"
#include <stdlib.h>
#include "internal/src/lite_log.h"

namespace mindspore::lite {
namespace {
constexpr size_t kMaxMallocSize = 2000 * 1024 * 1024;
constexpr int kBlockSize = 1024;
constexpr size_t kBlockLimit = (kBlockSize << (kBlockRange - 1));

int SizeToIndex(size_t size) {
  if (size > kBlockLimit) {
    return -1;
  }
  int index = 0;
  for (int i = 0; i < kBlockRange; ++i) {
    if ((size & (kBlockSize << i))) {
      index = i;
    }
  }
  if (size > (size_t)(kBlockSize << index)) {
    index += 1;
  }
  return index;
}

void PopMemNode(MemNode **head) {
  if (*head == nullptr) {
    return;
  }
  MemNode *next = (*head)->next_;
  (*head) = next;
  if (*head != nullptr) {
    (*head)->pre_ = nullptr;
  }
}

void PushMemNode(MemNode **head, MemNode *node) {
  if (node == nullptr) {
    return;
  }
  if (*head == nullptr) {
    *head = node;
    return;
  }
  (*head)->pre_ = node;
  node->next_ = *head;
  node->pre_ = nullptr;
  *head = node;
}

void RemoveMemNode(MemNode **head, MemNode *node) {
  if (node == nullptr) {
    return;
  }
  if ((*head) == node) {
    *head = node->next_;
    if (*head != nullptr) {
      (*head)->pre_ = nullptr;
    }
  } else {
    MemNode *node_pre = node->pre_;
    node_pre->next_ = node->next_;
    node->next_ = nullptr;
    node->pre_ = nullptr;
  }
}

void FreeNodesList(MemNode *head) {
  MemNode *node = head;
  while (node != nullptr) {
    MemNode *next = node->next_;
    free(node);
    node = next;
  }
}
}  // namespace

Allocator::Allocator() {
  for (int i = 0; i < kBlockRange; ++i) {
    allocated_list_[i] = nullptr;
    free_list_[i] = nullptr;
  }
}

Allocator::~Allocator() { Clear(); }

void Allocator::SetContext(const AllocatorContext &ctx) {
  lock_flag_ = ctx.lock_flag_;
}

void Allocator::Lock() {
  if (lock_flag_) {
    pthread_mutex_lock(&lock_);
  }
}

void Allocator::UnLock() {
  if (lock_flag_) {
    pthread_mutex_unlock(&lock_);
  }
}

void *Allocator::Malloc(size_t size) {
  if (size > kMaxMallocSize) {
    LITE_ERROR_LOG("MallocData out of max_size, size: %zd", size);
    return nullptr;
  }
  void *result = nullptr;
  int index = SizeToIndex(size);
  if (index < 0) {
    MemNode *node = (MemNode *)malloc(sizeof(MemNode) + size);
    if (node == nullptr) {
      LITE_ERROR_LOG("MallocData out of max_size, size: %zd", (size + sizeof(MemNode)));
      return result;
    }
    node->size_ = size;
    result = (char *)node + sizeof(MemNode);
    Lock();
    PushMemNode(&large_mem_list_, node);
    UnLock();
    return result;
  }
  Lock();
  size_t size_apply = (kBlockSize << index);
  if (free_list_[index] != nullptr) {
    MemNode *free_node = free_list_[index];
    PopMemNode(&free_list_[index]);
    PushMemNode(&allocated_list_[index], free_node);
    result = (char *)free_node + sizeof(MemNode);
    UnLock();
    return result;
  } else {
    MemNode *new_node = (MemNode *)malloc(sizeof(MemNode) + size_apply);
    if (new_node == nullptr) {
      UnLock();
      LITE_LOG_ERROR("malloc MemNode fail!");
      return nullptr;
    }
    new_node->size_ = size;
    PushMemNode(&allocated_list_[index], new_node);
    result = (char *)new_node + sizeof(MemNode);
    UnLock();
    return result;
  }
}

void Allocator::Free(void *buf) {
  if (buf == nullptr) {
    return;
  }

  MemNode *node = (MemNode *)((char *)buf - sizeof(MemNode));
  size_t buf_size = node->size_;
  Lock();
  if (buf_size > kBlockLimit) {
    RemoveMemNode(&large_mem_list_, node);
    free(node);
  } else {
    int index = SizeToIndex(buf_size);
    RemoveMemNode(&allocated_list_[index], node);
    PushMemNode(&free_list_[index], node);
  }
  UnLock();
}

size_t Allocator::GetTotalSize() {
  Lock();
  size_t total_size = 0;
  for (int i = 0; i < kBlockRange; ++i) {
    MemNode *node = allocated_list_[i];
    while (node != nullptr) {
      total_size += node->size_;
      node = node->next_;
    }

    node = free_list_[i];
    while (node != nullptr) {
      total_size += node->size_;
      node = node->next_;
    }
  }
  MemNode *node = large_mem_list_;
  while (node != nullptr) {
    total_size += node->size_;
    node = node->next_;
  }
  UnLock();
  return total_size;
}

void Allocator::Clear() {
  Lock();
  for (int i = 0; i < kBlockRange; ++i) {
    FreeNodesList(allocated_list_[i]);
    allocated_list_[i] = nullptr;

    FreeNodesList(free_list_[i]);
    free_list_[i] = nullptr;
  }
  FreeNodesList(large_mem_list_);
  UnLock();
}
}  // namespace mindspore::lite
