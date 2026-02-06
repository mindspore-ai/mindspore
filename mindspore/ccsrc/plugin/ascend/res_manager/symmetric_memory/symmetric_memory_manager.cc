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

#include "plugin/ascend/res_manager/symmetric_memory/symmetric_memory_manager.h"

#include <functional>
#include <string>
#include <memory>

#include "include/utils/common.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace ascend {
constexpr char kShmemLibName[] = "libshmem.so";
constexpr uint64_t kShmemSize = 1024 * 1024 * 1024;
constexpr char kShmemIpPort[] = "tcp://127.0.0.1:8769";

std::shared_ptr<SymmetricMemoryManager> &SymmetricMemoryManager::GetInstance() {
  static std::shared_ptr<SymmetricMemoryManager> instance = std::make_shared<SymmetricMemoryManager>();
  return instance;
}

void SymmetricMemoryManager::Finalize() {
  if (plugin_handle_ == nullptr) {
    return;
  }
#ifdef ENABLE_SYMMETRIC_MEMORY_KERNELS
  FinalizeShmem();
  FinalizePlugin();
#endif
}

void SymmetricMemoryManager::InitShmem(int my_rank, int n_ranks, size_t local_mem_size, const char *ip_port) {
#ifdef ENABLE_SYMMETRIC_MEMORY_KERNELS
  MS_LOG(INFO) << "Initing shmem, my_rank: " << my_rank << ", n_ranks: " << n_ranks
               << ", local_mem_size: " << local_mem_size << ", ip_port: " << ip_port;
  InitPlugin();
  shmem_init_attr_t *attributes;
  shmem_set_conf_store_tls_(false, nullptr, 0);
  if (shmem_set_attr_(my_rank, n_ranks, local_mem_size, ip_port, &attributes) != SHMEM_SUCCESS) {
    MS_LOG(EXCEPTION) << "Shmem SetAttr failed.";
  }
  if (shmem_init_attr_(attributes) != SHMEM_SUCCESS) {
    MS_LOG(EXCEPTION) << "Shmem InitAttr failed.";
  }
  MS_LOG(INFO) << "Init shmem success.";
#else
  MS_LOG(EXCEPTION) << "Symmetric memory is not enabled.";
#endif
}

void SymmetricMemoryManager::FinalizeShmem() {
  auto status = shmem_init_status_();
  if (status != 0) {
    shmem_finalize_();
    MS_LOG(INFO) << "Finalize shmem success.";
  }
}

void *SymmetricMemoryManager::AllocDeviceMemory(size_t size) {
  // lazy load shmem only when first time alloc symmetric memory
  if (!plugin_handle_) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    if (device_name != kAscendDevice) {
      MS_LOG(EXCEPTION) << "SymmetricMemoryManager only support Ascend device, but got device type: " << device_name;
    }
    auto rank_id = common::EnvHelper::GetInstance()->GetEnv("RANK_ID");
    auto rank_size = common::EnvHelper::GetInstance()->GetEnv("RANK_SIZE");
    if (rank_id == nullptr || rank_size == nullptr) {
      MS_LOG(EXCEPTION) << "RANK_ID or RANK_SIZE is empty";
    }
    int rank_id_int = std::stoi(rank_id);
    int rank_size_int = std::stoi(rank_size);
    uint64_t shmem_size_int = kShmemSize;
    std::string shmem_size_str = common::GetEnv("MS_SYMMETRIC_MEMORY_HEAP_SIZE");
    if (!shmem_size_str.empty()) {
      try {
        // str to uint64_t
        uint64_t temp_shmem_size = std::stoull(shmem_size_str);
        shmem_size_int = temp_shmem_size;
        MS_LOG(INFO) << "Manually set SYMMETRIC_MEMORY_HEAP_SIZE to " << shmem_size_int;
      } catch (const std::invalid_argument &e) {
        // handle invalid argument
        MS_LOG(WARNING) << "Invalid MS_SYMMETRIC_MEMORY_HEAP_SIZE value: " << shmem_size_str << ", using default value "
                        << shmem_size_int;
      } catch (const std::out_of_range &e) {
        // handle out of range
        MS_LOG(WARNING) << "MS_SYMMETRIC_MEMORY_HEAP_SIZE value " << shmem_size_str
                        << " is out of range, using default value " << shmem_size_int;
      }
    }
    const char *ip_port = GetShmemIpPort();
    InitShmem(rank_id_int, rank_size_int, shmem_size_int, ip_port);
  }
  MS_EXCEPTION_IF_NULL(plugin_handle_);
  auto ptr = shmem_malloc_(size);
  MS_LOG(DEBUG) << "Malloc symmetric memory, size: " << size << ", ptr: " << ptr;
  return ptr;
}

void SymmetricMemoryManager::FreeDeviceMemory(void *ptr) {
  MS_EXCEPTION_IF_NULL(plugin_handle_);
  MS_LOG(DEBUG) << "Free symmetric memory, ptr: " << ptr;
  shmem_free_(ptr);
}

const char *SymmetricMemoryManager::GetShmemIpPort() {
  // get MS_SCHED_HOST and MS_SCHED_PORT of scheduler from environment variables
  std::string sched_host = common::GetEnv("MS_SCHED_HOST");
  std::string sched_port_str = common::GetEnv("MS_SCHED_PORT");
  const char *ip_port = kShmemIpPort;
  // if MS_SCHED_HOST and MS_SCHED_PORT are set, use them to construct new ip_port
  if (!sched_host.empty() && !sched_port_str.empty()) {
    try {
      int sched_port = std::stoi(sched_port_str);
      int new_port = sched_port + 13;
      std::string new_ip_port = "tcp://" + sched_host + ":" + std::to_string(new_port);
      MS_LOG(INFO) << "Using environment variables for shmem IP port: " << new_ip_port;
      ip_port = new_ip_port.c_str();
    } catch (const std::invalid_argument &e) {
      MS_LOG(WARNING) << "Invalid MS_SCHED_PORT value: " << sched_port_str << ", using default: " << kShmemIpPort;
    } catch (const std::out_of_range &e) {
      MS_LOG(WARNING) << "MS_SCHED_PORT value " << sched_port_str
                      << " is out of range, using default: " << kShmemIpPort;
    }
  } else {
    MS_LOG(INFO) << "Using default shmem IP port: " << kShmemIpPort;
  }
  return ip_port;
}

void SymmetricMemoryManager::InitPlugin() {
  if (plugin_handle_ != nullptr) {
    return;
  }
#ifndef ENABLE_ASAN
  plugin_handle_ = dlopen(kShmemLibName, RTLD_DEEPBIND | RTLD_NOW | RTLD_LOCAL);
#else
  plugin_handle_ = dlopen(kShmemLibName, RTLD_NOW | RTLD_LOCAL);
#endif
  if (plugin_handle_ == nullptr) {
    MS_LOG(EXCEPTION) << "Dlopen " << kShmemLibName << " failed, result = " << GetDlErrorMsg();
  }
  shmem_init_status_ = DlsymFuncObj(shmem_init_status, plugin_handle_);
  MS_EXCEPTION_IF_NULL(shmem_init_status_);
  shmem_set_attr_ = DlsymFuncObj(shmem_set_attr, plugin_handle_);
  MS_EXCEPTION_IF_NULL(shmem_set_attr_);
  shmem_init_attr_ = DlsymFuncObj(shmem_init_attr, plugin_handle_);
  MS_EXCEPTION_IF_NULL(shmem_init_attr_);
  shmem_finalize_ = DlsymFuncObj(shmem_finalize, plugin_handle_);
  MS_EXCEPTION_IF_NULL(shmem_finalize_);
  shmemx_get_ffts_config_ = DlsymFuncObj(shmemx_get_ffts_config, plugin_handle_);
  MS_EXCEPTION_IF_NULL(shmemx_get_ffts_config_);
  shmem_set_conf_store_tls_ = DlsymFuncObj(shmem_set_conf_store_tls, plugin_handle_);
  MS_EXCEPTION_IF_NULL(shmem_set_conf_store_tls_);
  shmem_malloc_ = DlsymFuncObj(shmem_malloc, plugin_handle_);
  MS_EXCEPTION_IF_NULL(shmem_malloc_);
  shmem_free_ = DlsymFuncObj(shmem_free, plugin_handle_);
  MS_EXCEPTION_IF_NULL(shmem_free_);
  MS_LOG(INFO) << "Load " << kShmemLibName << " success.";
}
void SymmetricMemoryManager::FinalizePlugin() {
  if (plugin_handle_ == nullptr) {
    return;
  }
  shmem_init_status_ = nullptr;
  shmem_set_attr_ = nullptr;
  shmem_init_attr_ = nullptr;
  shmem_finalize_ = nullptr;
  shmemx_get_ffts_config_ = nullptr;
  shmem_set_conf_store_tls_ = nullptr;
  shmem_malloc_ = nullptr;
  shmem_free_ = nullptr;
  (void)dlclose(plugin_handle_);
  plugin_handle_ = nullptr;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
