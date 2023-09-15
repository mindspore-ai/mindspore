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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_NUMA_ADAPTER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_NUMA_ADAPTER_H_
#include <cstdint>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mindspore {
namespace numa {
struct bitmask {
  uint64_t size;
  uint64_t *maskp;
};

struct NUMAInterface {
  int (*numa_available)(void);
  int (*numa_num_configured_nodes)(void);
  int (*numa_num_task_cpus)();
  struct bitmask *(*numa_allocate_nodemask)(void);
  struct bitmask *(*numa_bitmask_clearall)(struct bitmask *);
  struct bitmask *(*numa_bitmask_setbit)(struct bitmask *, unsigned int);
  void (*numa_bind)(struct bitmask *);
  void (*numa_bitmask_free)(struct bitmask *);
  void *(*numa_alloc_onnode)(size_t size, int node);
  int64_t (*numa_node_size64)(int node, int64_t *freep);
  void (*numa_free)(void *start, size_t size);
};

struct MemoryInfo {
  int64_t total = 0;
  int64_t free = 0;
};

class NUMAAdapter {
 public:
  static std::shared_ptr<NUMAAdapter> GetInstance() {
    static std::shared_ptr<NUMAAdapter> const instance = std::make_shared<NUMAAdapter>();
    return instance;
  }

  NUMAAdapter();
  ~NUMAAdapter();
  inline bool Available() const { return available_; }
  void Bind(int node_id) const;
  void *Malloc(int node_id, size_t size) const;
  void Free(void *data, size_t size) const;
  int NodesNum() const;
  int CPUNum() const;
  std::vector<int> GetCPUList(int node_id);
  MemoryInfo GetNodeSize(int node_id) const;

 private:
  void *handle_;  // numa.so handle
  bool available_ = false;
  NUMAInterface numa_interfaces_;
  std::unordered_map<int, std::vector<int>> node_cpu_list_;
};
}  // namespace numa
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_NUMA_ADAPTER_H_
