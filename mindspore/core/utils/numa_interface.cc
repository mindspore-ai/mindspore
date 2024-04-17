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
#include "utils/numa_interface.h"

#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif

#include <dlfcn.h>
#include <cerrno>
#include <memory>
#include <mutex>
#include <vector>
#include "utils/log_adapter.h"

#define RETURN_STATUS_UNEXPECTED(_e)                                \
  do {                                                              \
    return Status(StatusCode::kCoreFailed, __LINE__, __FILE__, _e); \
  } while (false)

namespace mindspore {
namespace {
struct bitmask {
  uint64_t size;
  uint64_t *maskp;
};

std::weak_ptr<void> g_numa_lib_handle;
std::mutex g_numa_lib_handle_mutex;
}  // namespace

inline void *LoadLibrary(const char *name) {
  if (name == nullptr) {
    return nullptr;
  }
  auto handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
  return handle;
}

inline void *GetNumaAdapterFunc(void *handle, const char *name) {
  if (handle == nullptr) {
    MS_LOG(ERROR) << "The pointer[handle] is null.";
    return nullptr;
  }
  if (name == nullptr) {
    MS_LOG(ERROR) << "The pointer[name] is null.";
    return nullptr;
  }
  void *func = dlsym(handle, name);
  return func;
}

void ReleaseLibrary(void *handle) {
  if (handle != nullptr) {
    (void)dlclose(handle);
  }
}

std::shared_ptr<void> GetNumaAdapterHandle() {
  std::lock_guard<std::mutex> lock(g_numa_lib_handle_mutex);
  auto shared = g_numa_lib_handle.lock();
  if (shared != nullptr) {
    return shared;
  }
  void *handle = LoadLibrary("libnuma.so");
  shared = std::shared_ptr<void>(handle, ReleaseLibrary);
  g_numa_lib_handle = shared;
  return shared;
}

Status NumaBind(void *handle, const int32_t &rank_id) {
  if (handle == nullptr) {
    RETURN_STATUS_UNEXPECTED("Numa package not found.");
  }
  auto numa_max_node_func = GetNumaAdapterFunc(handle, "numa_max_node");
  if (numa_max_node_func == nullptr) {
    RETURN_STATUS_UNEXPECTED("Numa api: numa_max_node not found.");
  }
  auto numa_allocate_nodemask_func = GetNumaAdapterFunc(handle, "numa_allocate_nodemask");
  if (numa_allocate_nodemask_func == nullptr) {
    RETURN_STATUS_UNEXPECTED("Numa api: numa_allocate_nodemask not found.");
  }
  auto numa_bitmask_clearall_func = GetNumaAdapterFunc(handle, "numa_bitmask_clearall");
  if (numa_bitmask_clearall_func == nullptr) {
    RETURN_STATUS_UNEXPECTED("Numa api: numa_bitmask_clearall not found.");
  }
  auto numa_bitmask_setbit_func = GetNumaAdapterFunc(handle, "numa_bitmask_setbit");
  if (numa_bitmask_setbit_func == nullptr) {
    RETURN_STATUS_UNEXPECTED("Numa api: numa_bitmask_setbit not found.");
  }
  // fix: numa_bind failed with "set_mempolicy operation not permitted" on cloud env
  auto numa_run_on_node_mask_func = GetNumaAdapterFunc(handle, "numa_run_on_node_mask");
  if (numa_run_on_node_mask_func == nullptr) {
    RETURN_STATUS_UNEXPECTED("Numa api: numa_run_on_node_mask not found.");
  }
  auto set_mempolicy_func = GetNumaAdapterFunc(handle, "set_mempolicy");
  if (set_mempolicy_func == nullptr) {
    RETURN_STATUS_UNEXPECTED("Numa api: set_mempolicy not found.");
  }
  auto numa_set_membind_func = GetNumaAdapterFunc(handle, "numa_set_membind");
  if (numa_set_membind_func == nullptr) {
    RETURN_STATUS_UNEXPECTED("Numa api: numa_set_membind not found.");
  }
  auto numa_bitmask_free_func = GetNumaAdapterFunc(handle, "numa_bitmask_free");
  if (numa_bitmask_free_func == nullptr) {
    RETURN_STATUS_UNEXPECTED("Numa api: numa_bitmask_free not found.");
  }
  auto numa_max_node = (int (*)(void))(numa_max_node_func);
  auto numa_allocate_nodemask = (struct bitmask * (*)(void))(numa_allocate_nodemask_func);
  auto numa_bitmask_clearall = (struct bitmask * (*)(struct bitmask *))(numa_bitmask_clearall_func);
  auto numa_bitmask_setbit = (struct bitmask * (*)(struct bitmask *, unsigned int))(numa_bitmask_setbit_func);
  auto numa_run_on_node_mask = (int (*)(struct bitmask *))(numa_run_on_node_mask_func);
  auto set_mempolicy = (int (*)(int, const uint64_t *, uint64_t))(set_mempolicy_func);
  auto numa_set_membind = (void (*)(struct bitmask *))(numa_set_membind_func);
  auto numa_bitmask_free = (void (*)(struct bitmask *))(numa_bitmask_free_func);
  int numa_node_max_id = numa_max_node();
  if (numa_node_max_id < 0) {
    RETURN_STATUS_UNEXPECTED("Get numa max node failed.");
  }
  if (rank_id >= 0) {
    uint32_t numa_bind_id = static_cast<uint32_t>(rank_id % (numa_node_max_id + 1));
    auto bm = numa_allocate_nodemask();
    numa_bitmask_clearall(bm);
    numa_bitmask_setbit(bm, numa_bind_id);
    if (numa_run_on_node_mask(bm) < 0) {
      MS_LOG(WARNING) << "Try to bind numa id: " << numa_bind_id
                      << ", but execute numa_run_on_node_mask failed, errno: " << strerror(errno)
                      << ". Please use mindspore.dataset.config.set_numa_enable(False) to disable numa bind.";
      numa_bitmask_free(bm);
      return Status::OK();
    }
    if (set_mempolicy(MPOL_BIND, bm->maskp, bm->size + 1) < 0) {
      MS_LOG(WARNING) << "Try to bind numa id: " << numa_bind_id
                      << ", but execute set_mempolicy failed, errno: " << strerror(errno)
                      << ". Please use mindspore.dataset.config.set_numa_enable(False) to disable numa bind.";
      numa_bitmask_free(bm);
      return Status::OK();
    }
    numa_set_membind(bm);
    numa_bitmask_free(bm);
  } else {
    RETURN_STATUS_UNEXPECTED("Value error, rank_id is a negative value.");
  }
  return Status::OK();
}

#define DEFINE_NUMA_METHOD(handle, func_name, return_type, ...)     \
  auto func_name##_func = GetNumaAdapterFunc(handle, #func_name);   \
  if (func_name##_func == nullptr) {                                \
    RETURN_STATUS_UNEXPECTED("Numa api: numa_max_node not found."); \
  }                                                                 \
  auto func_name = (return_type(*)(__VA_ARGS__))(func_name##_func)

Status LoadNumaCpuInfo(void *handle, const int32_t rank_id, std::vector<int> *numa_cpus) {
  DEFINE_NUMA_METHOD(handle, numa_available, int);
  // Call numa_available first to make sure numa available.
  auto flag = numa_available();
  if (flag == -1) {
    // Here we do not care return value.
    return Status::OK();
  }

  DEFINE_NUMA_METHOD(handle, numa_max_node, int);
  DEFINE_NUMA_METHOD(handle, numa_num_task_cpus, int);
  DEFINE_NUMA_METHOD(handle, numa_bitmask_alloc, struct bitmask *, int);
  DEFINE_NUMA_METHOD(handle, numa_node_to_cpus, int, int, struct bitmask *);
  DEFINE_NUMA_METHOD(handle, numa_bitmask_isbitset, int, const struct bitmask *, unsigned int);
  DEFINE_NUMA_METHOD(handle, numa_bitmask_free, void, struct bitmask *);
  // Load cpus in current numa node.
  auto numa_node_max_id = numa_max_node();
  uint32_t numa_bind_id = static_cast<uint32_t>(rank_id % (numa_node_max_id + 1));
  MS_LOG(INFO) << "rank id : " << rank_id << ", numa bind id : " << numa_bind_id
               << ", numa_node_max_id : " << numa_node_max_id << ".";
  auto numa_cpu_num = numa_num_task_cpus();
  MS_LOG(DEBUG) << "Numa cpu num : " << numa_cpu_num << ".";
  // API numa_bitmask_alloc need min bitmask num 1024, or numa_node_to_cpus can not called twice.
  int bitmask_num = 1024;
  numa_cpu_num = std::max(numa_cpu_num, bitmask_num);
  auto numa_cpu_mask = numa_bitmask_alloc(numa_cpu_num);
  numa_node_to_cpus(numa_bind_id, numa_cpu_mask);
  MS_LOG(DEBUG) << "Numa numa_available call return : " << flag << ", numa_cpu_mask size : " << numa_cpu_mask->size
                << ".";
  for (int i = 0; i < numa_cpu_num; i++) {
    if (numa_bitmask_isbitset(numa_cpu_mask, i)) {
      (void)numa_cpus->emplace_back(i);
      MS_LOG(DEBUG) << "Numa cpus put : " << i << ".";
    }
  }
  numa_bitmask_free(numa_cpu_mask);
  return Status::OK();
}
}  // namespace mindspore
