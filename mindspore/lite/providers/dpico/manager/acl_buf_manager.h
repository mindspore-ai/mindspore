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

#ifndef MINDSPORE_LITE_PROVIDERS_DPICO_MANAGER_ACL_BUF_MANAGER_H
#define MINDSPORE_LITE_PROVIDERS_DPICO_MANAGER_ACL_BUF_MANAGER_H

#include "manager/acl_model_helper.h"
#include "include/errorcode.h"
#include "common/check_base.h"

namespace mindspore {
namespace lite {
class AclBufManager {
 public:
  static AclBufManager *GetInstance();
  const int &GetTaskBufSize() const { return task_buf_size_; }
  const int &GetWorkBufSize() const { return work_buf_size_; }
  void *GetTaskBufPtr();
  void *GetWorkBufPtr();
  int UpdateTaskBufSize(int task_buf_size);
  int UpdateWorkBufSize(int work_buf_size);

 private:
  AclBufManager() = default;
  ~AclBufManager();
  int task_buf_size_{0};
  int work_buf_size_{0};
  void *task_buf_ptr_{nullptr};
  void *work_buf_ptr_{nullptr};
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_PROVIDERS_DPICO_MANAGER_ACL_BUF_MANAGER_H
