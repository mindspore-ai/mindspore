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
#include "minddata/dataset/util/services.h"

#include <limits.h>
#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/syscall.h>
#else
#include <stdlib.h>
#endif
#include <unistd.h>
#include "minddata/dataset/engine/cache/cache_server.h"
#include "minddata/dataset/util/circular_pool.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
std::unique_ptr<Services> Services::instance_ = nullptr;
std::once_flag Services::init_instance_flag_;

#if !defined(_WIN32) && !defined(_WIN64)
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
#endif

std::string Services::GetUniqueID() {
  const std::string kStr = "abcdefghijklmnopqrstuvwxyz0123456789";
  std::mt19937 gen = GetRandomDevice();
  std::uniform_int_distribution<uint32_t> dist(0, kStr.size() - 1);
  char buffer[UNIQUEID_LEN];
  for (int i = 0; i < UNIQUEID_LEN; i++) {
    buffer[i] = kStr[dist(gen)];
  }
  return std::string(buffer, UNIQUEID_LEN);
}

TaskManager &Services::getTaskMgrInstance() {
  Services &sm = GetInstance();
  return *(static_cast<TaskManager *>(sm.sa_[kSlotTaskMgr_]));
}

CacheServer &Services::getCacheServer() {
  Services &sm = GetInstance();
  return *(static_cast<CacheServer *>(sm.sa_[kSlotCacheMgr_]));
}

Status Services::CreateAllInstances() {
  // In order, TaskMgr, BufferMgr
  Status rc;
  sa_[kSlotTaskMgr_] = new (&rc, pool_) TaskManager();
  RETURN_IF_NOT_OK(rc);
  rc = sa_[kSlotTaskMgr_]->ServiceStart();
  RETURN_IF_NOT_OK(rc);
  // TODO(jesse) : Get the parameters from config file. Right now spill to /tmp and spawn 3 workers
#if !defined(_WIN32) && !defined(_WIN64)
  sa_[kSlotCacheMgr_] = new (&rc, pool_) CacheServer("/tmp", 3);
  RETURN_IF_NOT_OK(rc);
  rc = sa_[kSlotCacheMgr_]->ServiceStart();
#else
  sa_[kSlotCacheMgr_] = nullptr;
#endif
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
    CacheServer *cs = static_cast<CacheServer *>(sa_[kSlotCacheMgr_]);
    if (cs != nullptr) {
      (void)cs->ServiceStop();
      cs->~CacheServer();
      pool_->Deallocate(cs);
    }
    TaskManager *tm = static_cast<TaskManager *>(sa_[kSlotTaskMgr_]);
    if (tm != nullptr) {
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
