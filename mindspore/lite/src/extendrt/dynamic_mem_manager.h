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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DYNAMIC_MEM_MANAGER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DYNAMIC_MEM_MANAGER_H_

#include <memory>
#include <vector>
#include <mutex>
#include <map>
#include <unordered_map>
#include <deque>
#include "src/extendrt/numa_adapter.h"

namespace mindspore {
struct Block {
  // used_ may be true when ref_count_ == 0
  bool used_ = false;
  void *data_ = nullptr;
  size_t size_;
  int64_t index_;
  int64_t ref_count_ = 0;
  int64_t pre_index_ = -1;
  int64_t next_index_ = -1;
};

class MemOperator {
 public:
  explicit MemOperator(int node_id);
  ~MemOperator();

  void *Malloc(size_t size);
  void Free(void *ptr);
  int SetRefCount(void *ptr, int ref_count);
  int IncRefCount(void *ptr, int ref_count);
  int DecRefCount(void *ptr, int ref_count);
  int RefCount(void *ptr);
  inline void set_node_id(int node_id) { node_id_ = node_id; }
  inline int node_id(void) const { return node_id_; }

 private:
  Block *GetBlock();
  void EraseFreeBlock(const int64_t index);
  void AddGarbageBlock(const int64_t index);
  void *Allocate(size_t rounded_size, int node_id, size_t *allocate_size);

 private:
  int node_id_ = -1;
  // all data blocks
  size_t block_count_ = 0;
  int64_t garbage_block_;
  std::shared_ptr<numa::NUMAAdapter> numa_instance_ = nullptr;
  std::mutex mutex_;
  std::vector<Block> blocks_;
  // key: data size, value: Block index
  std::multimap<size_t, int64_t> free_blocks_;
  // key: data addr, value: Block index
  std::unordered_map<void *, int64_t> datas_;
  std::unordered_map<void *, size_t> all_datas_;
};

class DynamicMemManager {
 public:
  DynamicMemManager() = default;
  ~DynamicMemManager() = default;

  std::shared_ptr<MemOperator> GetMemOperator(const int node_id);

 private:
  std::map<int, std::shared_ptr<MemOperator>> nodes_mem_;
  std::mutex mutex_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DYNAMIC_MEM_MANAGER_H_
