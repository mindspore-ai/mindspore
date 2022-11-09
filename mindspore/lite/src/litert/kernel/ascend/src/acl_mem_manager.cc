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
#include <memory>
#include <algorithm>
#include <map>
#include <string>
#include "acl/acl.h"
#include "src/common/log_adapter.h"

namespace mindspore::kernel {
namespace acl {
void AclMemManager::UpdateWorkspace(size_t work_size, size_t weight_size) {
  if (work_size > work_mem_info_.mem_size) {
    work_mem_info_.mem_size = work_size;
    MS_LOG(DEBUG) << "Update work_size = " << work_size << " successful.";
  }

  if (weight_size > weight_mem_info_.mem_size) {
    weight_mem_info_.mem_size = weight_size;
    MS_LOG(DEBUG) << "Update weight_size = " << weight_size << " successful.";
  }
}

STATUS AclMemManager::GetModelWorkMem(AclModelMemInfo *acl_work_mem_info) {
  std::unique_lock<std::mutex> acl_mtx(acl_mem_alloc_mutex_);
  if (work_mem_info_.mem_addr == nullptr) {
    if (work_mem_info_.mem_size == 0) {
      return lite::RET_ERROR;
    }
    auto acl_ret = aclrtMalloc(&work_mem_info_.mem_addr, work_mem_info_.mem_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (acl_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
      return lite::RET_ERROR;
    }
    MS_LOG(DEBUG) << "Malloc max work size is " << work_mem_info_.mem_size;
  }
  *acl_work_mem_info = work_mem_info_;
  return lite::RET_OK;
}

STATUS AclMemManager::GetModelWeightMem(AclModelMemInfo *acl_weight_mem_info) {
  std::unique_lock<std::mutex> acl_mtx(acl_mem_alloc_mutex_);
  if (weight_mem_info_.mem_addr == nullptr) {
    if (weight_mem_info_.mem_size == 0) {
      return lite::RET_ERROR;
    }
    auto acl_ret = aclrtMalloc(&weight_mem_info_.mem_addr, weight_mem_info_.mem_size, ACL_MEM_MALLOC_NORMAL_ONLY);
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
  if (work_mem_info_.mem_addr != nullptr) {
    (void)aclrtFree(work_mem_info_.mem_addr);
    work_mem_info_.mem_addr = nullptr;
    work_mem_info_.mem_size = 0;
  }
  if (weight_mem_info_.mem_addr != nullptr) {
    (void)aclrtFree(weight_mem_info_.mem_addr);
    weight_mem_info_.mem_addr = nullptr;
    weight_mem_info_.mem_size = 0;
  }
}
}  // namespace acl
}  // namespace mindspore::kernel
