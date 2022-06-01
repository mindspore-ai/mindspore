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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_RUNTIME_ALLOCATOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_RUNTIME_ALLOCATOR_H_

#include <memory>
#include <map>
#include <unordered_map>
#include "include/api/allocator.h"
#include "include/errorcode.h"
#include "src/tensor.h"

namespace mindspore {
class RuntimeAllocator : public Allocator {
 public:
  explicit RuntimeAllocator(size_t aligned_size = 32);
  ~RuntimeAllocator() override;

 public:
  void *Malloc(size_t size) override { return nullptr; }
  void Free(void *ptr) override { return; }
  int RefCount(void *ptr) override { return (ptr == nullptr) ? RUNTIME_REFCOUNT : 0; }
  int SetRefCount(void *ptr, int ref_count) override { return 0; }
  int IncRefCount(void *ptr, int ref_count) override { return 0; }
  int DecRefCount(void *ptr, int ref_count) override { return 0; }

 public:
  void SetDataOffset(lite::Tensor *tensor, size_t offset);
  void MallocTensorData(lite::Tensor *tensor);
  void FreeTensorData(lite::Tensor *tensor);
  void *MallocOptData();
  const std::unordered_map<lite::Tensor *, size_t> &GetOffsetMap() const { return offset_map_; }
  void Clear(AllocatorPtr default_allocator);

 private:
  size_t FindMinFree(size_t size);

 private:
  void *data_ = nullptr;
  size_t total_size_ = 0;
  std::unordered_map<lite::Tensor *, size_t> offset_map_;
  std::map<size_t, size_t> free_list_; /* offset, size */
  std::map<size_t, size_t> used_list_; /* offset, size */
};

using RuntimeAllocatorPtr = std::shared_ptr<RuntimeAllocator>;
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_RUNTIME_ALLOCATOR_H_
