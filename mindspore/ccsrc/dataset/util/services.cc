/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "dataset/util/services.h"

#include <limits.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <random>
#include "dataset/util/circular_pool.h"
#include "dataset/util/task_manager.h"

#define SLOT_TASK_MGR 0
namespace mindspore {
namespace dataset {
std::unique_ptr<Services> Services::instance_ = nullptr;
std::once_flag Services::init_instance_flag_;

std::string Services::GetUserName() {
  char user[LOGIN_NAME_MAX];
  (void)getlogin_r(user, sizeof(user));
  return std::string(user);
}

std::string Services::GetHostName() {
  char host[LOGIN_NAME_MAX];
  (void)gethostname(host, sizeof(host));
  return std::string(host);
}

int Services::GetLWP() { return syscall(SYS_gettid); }

std::string Services::GetUniqueID() {
  const std::string kStr = "abcdefghijklmnopqrstuvwxyz0123456789";
  std::mt19937 gen{std::random_device{"/dev/urandom"}()};
  std::uniform_int_distribution<> dist(0, kStr.size() - 1);
  char buffer[UNIQUEID_LEN];
  for (int i = 0; i < UNIQUEID_LEN; i++) {
    buffer[i] = kStr[dist(gen)];
  }
  return std::string(buffer, UNIQUEID_LEN);
}

TaskManager &Services::getTaskMgrInstance() {
  Services &sm = GetInstance();
  return *(static_cast<TaskManager *>(sm.sa_[SLOT_TASK_MGR]));
}

Status Services::CreateAllInstances() {
  // In order, TaskMgr, BufferMgr
  Status rc;
  sa_[SLOT_TASK_MGR] = new (&rc, pool_) TaskManager();
  RETURN_IF_NOT_OK(rc);
  rc = sa_[SLOT_TASK_MGR]->ServiceStart();
  return rc;
}

Services::Services() : pool_(nullptr), sa_{nullptr} {
  Status rc = CircularPool::CreateCircularPool(&pool_, -1, 16, true);  // each arena 16M
  if (rc.IsError()) {
    std::terminate();
  }
}

Services::~Services() noexcept {
  try {
    // In reverse order
    TaskManager *tm = static_cast<TaskManager *>(sa_[SLOT_TASK_MGR]);
    if (tm) {
      (void)tm->ServiceStop();
      tm->~TaskManager();
      pool_->Deallocate(tm);
    }
  } catch (const std::exception &e) {
    // Do nothing.
  }
}
}  // namespace dataset
}  // namespace mindspore
