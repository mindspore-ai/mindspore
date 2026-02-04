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
#include "plugin/ascend/res_manager/allocator/shared_memory_allocator.h"
#include <sys/shm.h>
#include <memory>
#include <string>

namespace mindspore {
namespace device {
namespace ascend {
std::shared_ptr<SharedMemoryAllocator> SharedMemoryAllocator::instance = nullptr;
std::shared_ptr<SharedMemoryAllocator> &SharedMemoryAllocator::getInstance() {
  static std::shared_ptr<SharedMemoryAllocator> instance = std::make_shared<SharedMemoryAllocator>();
  return instance;
}

void *SharedMemoryAllocator::Alloc(size_t size, uint32_t stream_id) {
  // set device id
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  AscendHalManager::GetInstance().SetContext(device_id);

  // check ascend
  if (!ms_context || ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice) {
    MS_LOG(WARNING) << "Only support Ascend backend!";
    return nullptr;
  }
  g_rmaDevModel = GetRmDevModel();

  // select
  void *ptr = nullptr;
  if (g_rmaDevModel == RmaDevModel::PCIE_TH_DEV) {
    ptr = AllocWithSharedMemory(size);
  } else if (g_rmaDevModel == RmaDevModel::SVM_MAP_DEV) {
    ptr = AllocWithHostMemory(size);
  } else {
    return nullptr;
  }

  // register
  void *registered_ptr = RegisterMem(ptr, size, device_id);
  if (!registered_ptr) {
    MS_LOG(WARNING) << "Allocation failed";
    return nullptr;
  }

  MS_LOG(INFO) << "Set up the mapping relationship from " << registered_ptr << " -> " << ptr;
  ptr_map_[registered_ptr] = ptr;
  return registered_ptr;
}

bool SharedMemoryAllocator::Free(void *ptr) {
  MS_LOG(INFO) << "Free tensor device ptr: " << ptr;

  if (!ptr) return true;
  g_rmaDevModel = GetRmDevModel();
  bool ret = false;

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  halHostUnregister(ptr, device_id);

  auto it = ptr_map_.find(ptr);
  if (it != ptr_map_.end()) {
    MS_LOG(INFO) << "Free tensor host ptr: " << it->second;

    if (g_rmaDevModel == RmaDevModel::PCIE_TH_DEV) {
      ret = FreeSharedMemory(it->second, device_id);
    } else if (g_rmaDevModel == RmaDevModel::SVM_MAP_DEV) {
      ret = FreeHostMemory(it->second);
    }

    ptr_map_.erase(it->first);
  }
  MS_LOG(INFO) << "Free memory" << (ret ? " succeeded" : " failed");
  return ret;
}

void *SharedMemoryAllocator::GetHostPtrByDevicePtr(void *devicePtr) {
  auto it = ptr_map_.find(devicePtr);
  if (it != ptr_map_.end()) {
    return it->second;
  }
  return nullptr;
}

uint32_t SharedMemoryAllocator::GetRegisterFlag(RmaDevModel mode) {
  switch (mode) {
    case RmaDevModel::SVM_MAP_DEV:
      return HOST_SVM_MAP_DEV;
    default:
      return HOST_MEM_MAP_DEV_PCIE_TH;
  }
}

void *SharedMemoryAllocator::RegisterMem(void *memory, uint64_t datalen, int deviceId) {
  std::lock_guard<std::mutex> lock(mutex_);
  uint32_t flag = GetRegisterFlag(g_rmaDevModel);
  void *svmmem = nullptr;
  drvError_t drvRet = DRV_ERROR_NOT_SUPPORT;
  drvRet = halHostRegister(memory, datalen, flag, deviceId, &svmmem);
  if (drvRet != DRV_ERROR_NONE) {
    if (g_rmaDevModel == RmaDevModel::PCIE_TH_DEV) {
      FreeSharedMemory(memory, deviceId);
    } else {
      FreeHostMemory(memory);
    }
    MS_LOG(WARNING) << "halHostRegister failed, drvRet error code: " << std::to_string(static_cast<int32_t>(drvRet));
  }
  return svmmem;
}

RmaDevModel SharedMemoryAllocator::GetRmDevModel() {
  const char *soc_name_c = CALL_ASCEND_API(aclrtGetSocName);
  std::string soc_name(soc_name_c ? soc_name_c : "unknown");
  if (soc_name.find("910B") != std::string::npos) {
    return RmaDevModel::PCIE_TH_DEV;
  } else if (soc_name.find("910_93") != std::string::npos) {
    return RmaDevModel::SVM_MAP_DEV;
  } else {
    MS_LOG(WARNING) << "Unsupported chip type";
  }
  return RmaDevModel::ERROR_CHIP_TYPE;
}

void *SharedMemoryAllocator::AllocWithSharedMemory(size_t size) {
  auto shmId_tmp = shmget(IPC_PRIVATE, size, IPC_CREAT | 0666);
  MS_LOG(INFO) << "Create new shmId: " << shmId_tmp;

  if (shmId_tmp == -1) {
    MS_LOG(WARNING) << "shmget failed: " << strerror(errno);
    return nullptr;
  }

  void *ptr = shmat(shmId_tmp, nullptr, 0);
  MS_LOG(INFO) << "Create new host shared ptr: " << ptr;

  if (ptr == reinterpret_cast<void *>(-1)) {
    shmctl(shmId_tmp, IPC_RMID, nullptr);
    MS_LOG(WARNING) << "shmat failed: " << strerror(errno);
    shmId_tmp = -1;
    return nullptr;
  }
  if (memset_s(ptr, size, 0, size) != EOK) {
    shmctl(shmId_tmp, IPC_RMID, nullptr);
    MS_LOG(WARNING) << "shmat memset failed";
  }
  shm_map_[ptr] = shmId_tmp;
  return ptr;
}

void *SharedMemoryAllocator::AllocWithHostMemory(size_t size) {
  void *ptr = nullptr;
  aclError ret = CALL_ASCEND_API(aclrtMallocHost, &ptr, size);
  if (ret != ACL_ERROR_NONE) {
    return nullptr;
  }
  aclError ret_memset = CALL_ASCEND_API(aclrtMemset, &ptr, size, 0, size);
  if (ret_memset != ACL_ERROR_NONE) {
    FreeHostMemory(ptr);
    return nullptr;
  }
  return ptr;
}
bool SharedMemoryAllocator::FreeSharedMemory(void *ptr, int deviceId) {
  int shmId = -1;
  if (!GetAndRemoveShmId(ptr, &shmId)) {
    return false;
  }
  if (shmdt(ptr) == -1) {
    return false;
  }
  shmctl(shmId, IPC_RMID, nullptr);
  return true;
}

bool SharedMemoryAllocator::FreeHostMemory(void *ptr) {
  aclError ret = CALL_ASCEND_API(aclrtFreeHost, ptr);
  if (ret != ACL_ERROR_NONE) {
    return false;
  }
  return true;
}

bool SharedMemoryAllocator::GetAndRemoveShmId(void *ptr, int *shmId) {
  auto it = shm_map_.find(ptr);
  if (it == shm_map_.end()) {
    MS_LOG(WARNING) << "Unknown shared memory host pointer: " << ptr;
    return false;
  }
  *shmId = it->second;
  shm_map_.erase(it);
  return true;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
