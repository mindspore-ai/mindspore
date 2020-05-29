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
#ifndef DATASET_UTIL_SERVICES_H_
#define DATASET_UTIL_SERVICES_H_

#include <memory>
#include <mutex>
#include <string>
#include "dataset/util/memory_pool.h"
#include "dataset/util/allocator.h"
#include "dataset/util/service.h"

#define UNIQUEID_LEN 36
namespace mindspore {
namespace dataset {
class TaskManager;

class Services {
 public:
  static Status CreateInstance() {
    std::call_once(init_instance_flag_, [&]() -> Status {
      instance_.reset(new Services());
      return (instance_->CreateAllInstances());
    });

    if (instance_ == nullptr) {
      instance_.reset(new Services());
      return (instance_->CreateAllInstances());
    }

    return Status::OK();
  }

  static Services &GetInstance() {
    if (instance_ == nullptr) {
      if (!CreateInstance()) {
        std::terminate();
      }
    }
    return *instance_;
  }

  Services(const Services &) = delete;

  Services &operator=(const Services &) = delete;

  ~Services() noexcept;

  static TaskManager &getTaskMgrInstance();

  std::shared_ptr<MemoryPool> GetServiceMemPool() { return pool_; }

#if !defined(_WIN32) && !defined(_WIN64)
  static std::string GetUserName();

  static std::string GetHostName();

  static int GetLWP();
#endif

  static std::string GetUniqueID();

  template <typename T>
  static Allocator<T> GetAllocator() {
    return Allocator<T>(Services::GetInstance().GetServiceMemPool());
  }

 private:
  static std::once_flag init_instance_flag_;
  static std::unique_ptr<Services> instance_;
  // A small pool used for small objects that last until the
  // Services Manager shuts down. Used by all sub-services.
  std::shared_ptr<MemoryPool> pool_;
  // We use pointers here instead of unique_ptr because we
  // want to have ultimate control on the order of
  // construction and destruction.
  static constexpr int kNumServices_ = 1;
  Service *sa_[kNumServices_];

  Services();

  Status CreateAllInstances();
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_UTIL_SERVICES_H_
