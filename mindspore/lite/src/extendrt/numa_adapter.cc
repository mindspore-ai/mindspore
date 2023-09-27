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

#include "src/extendrt/numa_adapter.h"
#include <dlfcn.h>
#include <fstream>
#include <string>
#include "src/common/log_adapter.h"
#include "src/common/common.h"
#include "src/common/utils.h"

namespace mindspore {
namespace numa {
namespace {
static constexpr auto kNodeBase = "/sys/devices/system/node/node";
constexpr int kBase = 10;
}  // namespace

NUMAAdapter::NUMAAdapter() {
  available_ = false;
  handle_ = dlopen("libnuma.so", RTLD_LAZY | RTLD_LOCAL);
  if (handle_ == nullptr) {
    MS_LOG(WARNING) << "Open libnuma.so failed!try libnuma.so.1 again.";
    handle_ = dlopen("libnuma.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (handle_ == nullptr) {
      MS_LOG(WARNING) << "Open numa so failed!";
      return;
    }
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

void NUMAAdapter::Bind(int node_id) const {
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

void *NUMAAdapter::Malloc(int node_id, size_t size) const {
  if (!Available() || node_id < 0) {
    return nullptr;
  }
  return numa_interfaces_.numa_alloc_onnode(size, node_id);
}

void NUMAAdapter::Free(void *data, size_t size) const {
  if (!Available() || data == nullptr) {
    return;
  }
  numa_interfaces_.numa_free(data, size);
}

int NUMAAdapter::NodesNum() const {
  if (!Available()) {
    return 0;
  }
  return numa_interfaces_.numa_num_configured_nodes();
}

int NUMAAdapter::CPUNum() const {
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
  auto iter = node_cpu_list_.find(node_id);
  if (iter != node_cpu_list_.end()) {
    return iter->second;
  }
  static constexpr auto kVectorDefaultSize = 32;
  cpu_list.reserve(kVectorDefaultSize);
  std::string cpu_list_file = kNodeBase + std::to_string(node_id) + "/cpulist";
  std::ifstream cpu_list_ifs(cpu_list_file);
  if (!cpu_list_ifs.good()) {
    MS_LOG(ERROR) << "file: " << cpu_list_file << " is not exist";
    return cpu_list;
  }
  if (!cpu_list_ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << cpu_list_file << " open failed";
    return cpu_list;
  }
  std::string line;
  if (!std::getline(cpu_list_ifs, line)) {
    MS_LOG(ERROR) << cpu_list_file << " getline failed!";
    return cpu_list;
  }
  lite::Trim(&line);
  auto cpu_lists = lite::StrSplit(line, ",");
  for (auto &&item : cpu_lists) {
    auto cpu_range = lite::StrSplit(item, "-");
    static constexpr size_t kMaxRangeNum = 2;
    if (cpu_range.size() != kMaxRangeNum) {
      continue;
    }
    int begin = static_cast<int>(strtol(cpu_range[0].c_str(), nullptr, kBase));
    int end = static_cast<int>(strtol(cpu_range[1].c_str(), nullptr, kBase));
    for (int j = begin; j <= end; ++j) {
      cpu_list.emplace_back(j);
    }
  }
  cpu_list_ifs.close();
  node_cpu_list_[node_id] = cpu_list;
  return cpu_list;
}

MemoryInfo NUMAAdapter::GetNodeSize(int node_id) const {
  MemoryInfo mem_info;
  if (!Available() || node_id < 0) {
    return mem_info;
  }
  mem_info.total = numa_interfaces_.numa_node_size64(node_id, &mem_info.free);
  return mem_info;
}

NUMAAdapter::~NUMAAdapter() {
  MS_LOG(DEBUG) << "~NUMAAdapter() begin.";
  if (handle_ == nullptr) {
    return;
  }
  (void)dlclose(handle_);
  handle_ = nullptr;
  MS_LOG(DEBUG) << "~NUMAAdapter() end.";
}
}  // namespace numa
}  // namespace mindspore
