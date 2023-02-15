/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DYNAMIC_MEM_ALLOCATOR_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DYNAMIC_MEM_ALLOCATOR_H_

#include <mutex>
#include <map>
#include <memory>
#include <unordered_map>
#include "include/api/allocator.h"
#include "src/extendrt/dynamic_mem_manager.h"

namespace mindspore {
class DynamicMemAllocator : public Allocator {
 public:
  explicit DynamicMemAllocator(int node_id);
  virtual ~DynamicMemAllocator() = default;
  void *Malloc(size_t size) override;
  void Free(void *ptr) override;
  int RefCount(void *ptr) override;
  int SetRefCount(void *ptr, int ref_count) override;
  int IncRefCount(void *ptr, int ref_count) override;
  int DecRefCount(void *ptr, int ref_count) override;

 private:
  std::mutex allocator_mutex_;
  std::shared_ptr<DynamicMemManager> mem_manager_ = nullptr;
  std::shared_ptr<MemOperator> mem_oper_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DYNAMIC_MEM_ALLOCATOR_H_
