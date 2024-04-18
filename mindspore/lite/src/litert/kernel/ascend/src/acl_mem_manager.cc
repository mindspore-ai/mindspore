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

#include "src/litert/kernel/ascend/src/acl_mem_manager.h"
#include <utility>
#include <memory>
#include <algorithm>
#include <map>
#include <string>
#include "src/common/log_adapter.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore::kernel {
namespace acl {
STATUS AclMemManager::UpdateWorkspace(size_t work_size, size_t weight_size, int32_t device_id) {
  auto it = work_mem_info_map_.find(device_id);
  if (it == work_mem_info_map_.end()) {
    AclModelMemInfo new_work_mem = {nullptr, 0};
    work_mem_info_map_.insert(std::make_pair(device_id, std::make_pair(new_work_mem, false)));
  } else if (it->second.second == true) {
    MS_LOG(ERROR) << "Device " << device_id << " has alloc memory!";
    return lite::RET_ERROR;
  }

  it = work_mem_info_map_.find(device_id);
  if (it == work_mem_info_map_.end()) {
    MS_LOG(ERROR) << "Get mem failed!";
    return lite::RET_ERROR;
  }

  if (work_size > it->second.first.mem_size) {
    it->second.first.mem_size = work_size;
    MS_LOG(DEBUG) << "Update work_size = " << it->second.first.mem_size << " successful.";
  }

  if (weight_size > weight_mem_info_.mem_size) {
    weight_mem_info_.mem_size = weight_size;
    MS_LOG(DEBUG) << "Update weight_size = " << weight_size << " successful.";
  }
  return lite::RET_OK;
}

STATUS AclMemManager::GetModelWorkMem(AclModelMemInfo *acl_work_mem_info, int32_t device_id) {
  std::unique_lock<std::mutex> acl_mtx(acl_mem_alloc_mutex_);

  auto it = work_mem_info_map_.find(device_id);
  if (it == work_mem_info_map_.end()) {
    MS_LOG(ERROR) << "Get work mem failed!";
    return lite::RET_ERROR;
  }
  it->second.second = true;

  if (it->second.first.mem_addr == nullptr) {
    if (it->second.first.mem_size == 0) {
      return lite::RET_ERROR;
    }
    auto acl_ret =
      CALL_ASCEND_API(aclrtMalloc, &(it->second.first.mem_addr), it->second.first.mem_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
      return lite::RET_ERROR;
    }
    MS_LOG(DEBUG) << "Malloc max work size is " << it->second.first.mem_size;
  }
  *acl_work_mem_info = it->second.first;
  return lite::RET_OK;
}

STATUS AclMemManager::GetModelWeightMem(AclModelMemInfo *acl_weight_mem_info) {
  std::unique_lock<std::mutex> acl_mtx(acl_mem_alloc_mutex_);
  if (weight_mem_info_.mem_addr == nullptr) {
    if (weight_mem_info_.mem_size == 0) {
      return lite::RET_ERROR;
    }
    auto acl_ret =
      CALL_ASCEND_API(aclrtMalloc, &weight_mem_info_.mem_addr, weight_mem_info_.mem_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
      return lite::RET_ERROR;
    }
    MS_LOG(DEBUG) << "Malloc max weight size is " << weight_mem_info_.mem_size;
  }
  *acl_weight_mem_info = weight_mem_info_;
  return lite::RET_OK;
}

AclMemManager::~AclMemManager() {
  for (auto &mem_info_pair : work_mem_info_map_) {
    if (mem_info_pair.second.first.mem_addr != nullptr) {
      (void)CALL_ASCEND_API(aclrtFree, mem_info_pair.second.first.mem_addr);
      mem_info_pair.second.first.mem_addr = nullptr;
      mem_info_pair.second.first.mem_size = 0;
    }
  }
  if (weight_mem_info_.mem_addr != nullptr) {
    (void)CALL_ASCEND_API(aclrtFree, weight_mem_info_.mem_addr);
    weight_mem_info_.mem_addr = nullptr;
    weight_mem_info_.mem_size = 0;
  }
}
}  // namespace acl
}  // namespace mindspore::kernel
