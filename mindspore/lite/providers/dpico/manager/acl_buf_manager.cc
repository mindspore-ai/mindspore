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

#include "manager/acl_buf_manager.h"
#include "manager/acl_model_helper.h"
#include "common/op_attr.h"

namespace mindspore {
namespace lite {
AclBufManager *AclBufManager::GetInstance() {
  static AclBufManager instance;
  return &instance;
}

void *AclBufManager::GetTaskBufPtr() {
  if (task_buf_ptr_ == nullptr) {
    int ret = AclMalloc(&task_buf_ptr_, task_buf_size_);
    MS_LOG(INFO) << "task_buf_size is  " << task_buf_size_;
    MS_CHECK_TRUE_MSG(ret == RET_OK, nullptr, "svp acl rt malloc task buf failed.");
  }
  return task_buf_ptr_;
}

void *AclBufManager::GetWorkBufPtr() {
  if (work_buf_ptr_ == nullptr) {
    int ret = AclMalloc(&work_buf_ptr_, work_buf_size_);
    MS_LOG(INFO) << "work_buf_size is  " << work_buf_size_;
    MS_CHECK_TRUE_MSG(ret == RET_OK, nullptr, "svp acl rt malloc work buf failed.");
  }
  return work_buf_ptr_;
}

int AclBufManager::UpdateTaskBufSize(int task_buf_size) {
  if (task_buf_size <= task_buf_size_) {
    MS_LOG(INFO) << "new task_buf_size is " << task_buf_size << ", old task_buf_size_ is " << task_buf_size_
                 << ", no need to update task buf";
    return RET_OK;
  }
  MS_LOG(INFO) << "new task_buf_size is " << task_buf_size << ", old task_buf_size_ is " << task_buf_size_
               << ", will update task buf";
  task_buf_size_ = task_buf_size;
  return RET_OK;
}

int AclBufManager::UpdateWorkBufSize(int work_buf_size) {
  if (work_buf_size <= work_buf_size_) {
    MS_LOG(INFO) << "new work_size is " << work_buf_size << ", old work_buf_size_ is " << work_buf_size_
                 << ", no need to update work buf";
    return RET_OK;
  }
  MS_LOG(INFO) << "new work_buf_size is " << work_buf_size << ", old work_buf_size_ is " << work_buf_size_
               << ", will update work buf";
  work_buf_size_ = work_buf_size;
  return RET_OK;
}

AclBufManager::~AclBufManager() {
  if (task_buf_ptr_ != nullptr) {
    int ret = AclFree(&task_buf_ptr_);
    MS_CHECK_TRUE_MSG_VOID(ret == RET_OK, "acl free task_buf_ptr_ failed.");
  }
  if (work_buf_ptr_ != nullptr) {
    int ret = AclFree(&work_buf_ptr_);
    MS_CHECK_TRUE_MSG_VOID(ret == RET_OK, "acl free work_buf_ptr_ failed.");
  }
}
}  // namespace lite
}  // namespace mindspore
