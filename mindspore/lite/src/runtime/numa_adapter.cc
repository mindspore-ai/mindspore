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
#include "src/runtime/numa_adapter.h"
#include <dlfcn.h>
#include "src/common/log_adapter.h"
#include "src/common/common.h"

namespace mindspore {
namespace numa {
namespace {
static constexpr int kSuccess = 0;
static constexpr int kBitsPerByte = 8;
static constexpr auto kBitsPerMask = static_cast<int>(sizeof(uint64_t) * kBitsPerByte);
}  // namespace

NUMAAdapter::NUMAAdapter() {
  available_ = false;
  handle_ = dlopen("libnuma.so.1.0.0", RTLD_LAZY | RTLD_LOCAL);
  if (handle_ == nullptr) {
    MS_LOG(WARNING) << "Does not support NUMA.";
    return;
  }

  numa_interfaces_.numa_available = reinterpret_cast<int (*)(void)>(dlsym(handle_, "numa_available"));
  if (MS_UNLIKELY(numa_interfaces_.numa_available == nullptr)) {
    MS_LOG(ERROR) << "numa_available not found!";
  }
  if (numa_interfaces_.numa_available() < 0) {
    MS_LOG(ERROR) << "numa is not available!";
    (void)dlclose(handle_);
    handle_ = nullptr;
    return;
  }
  available_ = true;
  numa_interfaces_.numa_num_configured_nodes =
    reinterpret_cast<int (*)(void)>(dlsym(handle_, "numa_num_configured_nodes"));
  if (MS_UNLIKELY(numa_interfaces_.numa_num_configured_nodes == nullptr)) {
    MS_LOG(ERROR) << "numa_num_configured_nodes not found!";
    available_ = false;
  }
  numa_interfaces_.numa_num_task_cpus = reinterpret_cast<int (*)(void)>(dlsym(handle_, "numa_num_task_cpus"));
  if (MS_UNLIKELY(numa_interfaces_.numa_num_task_cpus == nullptr)) {
    MS_LOG(ERROR) << "numa_num_task_cpus not found!";
    available_ = false;
  }
  numa_interfaces_.numa_node_to_cpus =
    reinterpret_cast<int (*)(int node, struct bitmask *mask)>(dlsym(handle_, "numa_node_to_cpus"));
  if (MS_UNLIKELY(numa_interfaces_.numa_node_to_cpus == nullptr)) {
    MS_LOG(ERROR) << "numa_node_to_cpus not found!";
    available_ = false;
  }
  numa_interfaces_.numa_allocate_nodemask =
    reinterpret_cast<struct bitmask *(*)(void)>(dlsym(handle_, "numa_allocate_nodemask"));
  if (MS_UNLIKELY(numa_interfaces_.numa_allocate_nodemask == nullptr)) {
    MS_LOG(ERROR) << "numa_allocate_nodemask not found!";
    available_ = false;
  }
  numa_interfaces_.numa_bitmask_clearall =
    reinterpret_cast<struct bitmask *(*)(struct bitmask *)>(dlsym(handle_, "numa_bitmask_clearall"));
  if (MS_UNLIKELY(numa_interfaces_.numa_bitmask_clearall == nullptr)) {
    MS_LOG(ERROR) << "numa_bitmask_clearall not found!";
    available_ = false;
  }
  numa_interfaces_.numa_bitmask_setbit =
    reinterpret_cast<struct bitmask *(*)(struct bitmask *, unsigned int)>(dlsym(handle_, "numa_bitmask_setbit"));
  if (MS_UNLIKELY(numa_interfaces_.numa_bitmask_setbit == nullptr)) {
    MS_LOG(ERROR) << "numa_bitmask_setbit not found!";
    available_ = false;
  }
  numa_interfaces_.numa_bind = reinterpret_cast<void (*)(struct bitmask *)>(dlsym(handle_, "numa_bind"));
  if (MS_UNLIKELY(numa_interfaces_.numa_bind == nullptr)) {
    MS_LOG(ERROR) << "numa_bind not found!";
    available_ = false;
  }
  numa_interfaces_.numa_bitmask_free =
    reinterpret_cast<void (*)(struct bitmask *)>(dlsym(handle_, "numa_bitmask_free"));
  if (MS_UNLIKELY(numa_interfaces_.numa_bitmask_free == nullptr)) {
    MS_LOG(ERROR) << "numa_bitmask_free not found!";
    available_ = false;
  }
  numa_interfaces_.numa_alloc_onnode =
    reinterpret_cast<void *(*)(size_t size, int node)>(dlsym(handle_, "numa_alloc_onnode"));
  if (MS_UNLIKELY(numa_interfaces_.numa_alloc_onnode == nullptr)) {
    MS_LOG(ERROR) << "numa_bitmask_free not found!";
    available_ = false;
  }
  numa_interfaces_.numa_node_size64 =
    reinterpret_cast<int64_t (*)(int node, int64_t *freep)>(dlsym(handle_, "numa_node_size64"));
  if (MS_UNLIKELY(numa_interfaces_.numa_node_size64 == nullptr)) {
    MS_LOG(ERROR) << "numa_node_size64 not found!";
    available_ = false;
  }
  numa_interfaces_.numa_free = reinterpret_cast<void (*)(void *start, size_t size)>(dlsym(handle_, "numa_free"));
  if (MS_UNLIKELY(numa_interfaces_.numa_free == nullptr)) {
    MS_LOG(ERROR) << "numa_free not found!";
    available_ = false;
  }
  if (!available_) {
    (void)dlclose(handle_);
    handle_ = nullptr;
    return;
  }
}

void NUMAAdapter::Bind(int node_id) {
  if (!Available() || node_id < 0) {
    return;
  }
  auto bitmask = numa_interfaces_.numa_allocate_nodemask();
  if (MS_UNLIKELY(bitmask == nullptr)) {
    MS_LOG(ERROR) << "bind numa_node " << node_id << " failed!";
    return;
  }
  (void)numa_interfaces_.numa_bitmask_setbit(bitmask, node_id);
  numa_interfaces_.numa_bind(bitmask);
  numa_interfaces_.numa_bitmask_free(bitmask);
}

void *NUMAAdapter::Malloc(int node_id, size_t size) {
  if (!Available() || node_id < 0) {
    return nullptr;
  }
  return numa_interfaces_.numa_alloc_onnode(size, node_id);
}

void NUMAAdapter::Free(void *data, size_t size) {
  if (!Available() || data == nullptr) {
    return;
  }
  numa_interfaces_.numa_free(data, size);
}

int NUMAAdapter::NodesNum() {
  if (!Available()) {
    return 0;
  }
  return numa_interfaces_.numa_num_configured_nodes();
}

int NUMAAdapter::CPUNum() {
  if (!Available()) {
    return 0;
  }
  return numa_interfaces_.numa_num_task_cpus();
}

std::vector<int> NUMAAdapter::GetCPUList(int node_id) {
  std::vector<int> cpu_list;
  if (!Available() || node_id < 0) {
    return cpu_list;
  }
  struct bitmask *nodemask = numa_interfaces_.numa_allocate_nodemask();
  if (nodemask == nullptr) {
    MS_LOG(ERROR) << "allocate nodemask failed!";
    return cpu_list;
  }
  auto ret = numa_interfaces_.numa_node_to_cpus(node_id, nodemask);
  if (ret != kSuccess || nodemask->maskp == nullptr) {
    MS_LOG(ERROR) << "numa_node_to_cpus failed!ret = " << ret;
    return cpu_list;
  }
  int cpu_num = numa_interfaces_.numa_num_task_cpus();
  if (MS_UNLIKELY(cpu_num < 0)) {
    MS_LOG(ERROR) << "numa_num_task_cpus return " << cpu_num;
    return cpu_list;
  }
  int index = 0;
  int maskp_index = 0;
  auto maskp = nodemask->maskp;
  do {
    if (MS_UNLIKELY(maskp == nullptr)) {
      MS_LOG(ERROR) << "maskp is nullptr!";
      break;
    }
    auto mask = *(maskp);
    int step = static_cast<int>(maskp_index * kBitsPerMask);
    for (int i = 0; i < kBitsPerMask; ++i) {
      if (mask & 1) {
        cpu_list.emplace_back(i + step);
      }
      mask >>= 1;
    }
    index += kBitsPerMask;
    if (index >= cpu_num) {
      break;
    }
    maskp = nodemask->maskp + 1;
    ++maskp_index;
  } while (true);

  numa_interfaces_.numa_bitmask_free(nodemask);
  return cpu_list;
}

MemoryInfo NUMAAdapter::GetNodeSize(int node_id) {
  MemoryInfo mem_info;
  if (!Available() || node_id < 0) {
    return mem_info;
  }
  mem_info.total = numa_interfaces_.numa_node_size64(node_id, &mem_info.free);
  return mem_info;
}

NUMAAdapter::~NUMAAdapter() {
  if (handle_ == nullptr) {
    return;
  }
  (void)dlclose(handle_);
  handle_ = nullptr;
}
}  // namespace numa
}  // namespace mindspore
