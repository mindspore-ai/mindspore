/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_SYMMETRIC_MEMORY_MANAGER_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_SYMMETRIC_MEMORY_MANAGER_H_

#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "device_address/device_address.h"
#include "plugin/ascend/res_manager/symmetric_memory/symmetric_memory_plugin.h"
#include "plugin/ascend/res_manager/visible.h"

namespace mindspore {
namespace device {
namespace ascend {
class SymmetricMemoryManager {
 public:
  static std::shared_ptr<SymmetricMemoryManager> &GetInstance();
  void Finalize();
  SymmetricMemoryManager() = default;
  ~SymmetricMemoryManager() = default;

  void InitShmem(int my_rank, int n_ranks, size_t local_mem_size, const char *ip_port);
  void FinalizeShmem();

  // Device memory
  void *AllocDeviceMemory(size_t size);
  void FreeDeviceMemory(void *ptr);

 private:
  struct compare {
    bool operator()(const DeviceAddressPtr &l, const DeviceAddressPtr &r) const { return l->GetSize() < r->GetSize(); }
  };
  const char *GetShmemIpPort();
  void InitPlugin();
  void FinalizePlugin();
  void *plugin_handle_ = nullptr;
  shmem_init_statusFunObj shmem_init_status_ = nullptr;
  shmem_set_attrFunObj shmem_set_attr_ = nullptr;
  shmem_init_attrFunObj shmem_init_attr_ = nullptr;
  shmem_finalizeFunObj shmem_finalize_ = nullptr;
  shmemx_get_ffts_configFunObj shmemx_get_ffts_config_ = nullptr;
  shmem_set_conf_store_tlsFunObj shmem_set_conf_store_tls_ = nullptr;

  // shmem heap
  shmem_mallocFunObj shmem_malloc_ = nullptr;
  shmem_freeFunObj shmem_free_ = nullptr;
};
using SymmetricMemoryManagerPtr = std::shared_ptr<SymmetricMemoryManager>;
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_SYMMETRIC_MEMORY_MANAGER_H_
