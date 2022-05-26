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
#include <dlfcn.h>
#include <memory>
#include <mutex>
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
  auto numa_bind_func = GetNumaAdapterFunc(handle, "numa_bind");
  if (numa_bind_func == nullptr) {
    RETURN_STATUS_UNEXPECTED("Numa api: numa_bind not found.");
  }
  auto numa_bitmask_free_func = GetNumaAdapterFunc(handle, "numa_bitmask_free");
  if (numa_bitmask_free_func == nullptr) {
    RETURN_STATUS_UNEXPECTED("Numa api: numa_bitmask_free not found.");
  }
  auto numa_max_node = (int (*)(void))(numa_max_node_func);
  auto numa_allocate_nodemask = (struct bitmask * (*)(void))(numa_allocate_nodemask_func);
  auto numa_bitmask_clearall = (struct bitmask * (*)(struct bitmask *))(numa_bitmask_clearall_func);
  auto numa_bitmask_setbit = (struct bitmask * (*)(struct bitmask *, unsigned int))(numa_bitmask_setbit_func);
  auto numa_bind = (void (*)(struct bitmask *))(numa_bind_func);
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
    numa_bind(bm);
    numa_bitmask_free(bm);
  } else {
    RETURN_STATUS_UNEXPECTED("Value error, rank_id is a negative value.");
  }
  return Status::OK();
}
}  // namespace mindspore
