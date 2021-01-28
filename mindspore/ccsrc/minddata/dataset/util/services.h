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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SERVICES_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SERVICES_H_

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "minddata/dataset/util/memory_pool.h"
#include "minddata/dataset/util/allocator.h"
#include "minddata/dataset/util/service.h"

#define UNIQUEID_LEN 36
#define UNIQUEID_LIST_LIMITS 1024
#define UNIQUEID_HALF_INDEX ((UNIQUEID_LIST_LIMITS) / 2)
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

  std::shared_ptr<MemoryPool> GetServiceMemPool() { return pool_; }

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  static std::string GetUserName();

  static std::string GetHostName();

  static int GetLWP();
#endif

  static std::string GetUniqueID();

  template <typename T>
  static Allocator<T> GetAllocator() {
    return Allocator<T>(Services::GetInstance().GetServiceMemPool());
  }

  /// \brief Add a new service to the start up list.
  /// \tparam T Class that implements Service
  /// \return Status object and where the service is located in the hook_ list
  template <typename T, typename... Args>
  Status AddHook(T **out, Args &&... args) {
    RETURN_UNEXPECTED_IF_NULL(out);
    try {
      (*out) = new T(std::forward<Args>(args)...);
      std::unique_ptr<T> svc(*out);
      hook_.push_back(std::move(svc));
    } catch (const std::bad_alloc &e) {
      return Status(StatusCode::kMDOutOfMemory);
    }
    return Status::OK();
  }

 private:
  static std::once_flag init_instance_flag_;
  static std::unique_ptr<Services> instance_;
  static std::map<std::string, uint64_t> unique_id_list_;
  static uint64_t unique_id_count_;
  static std::mutex unique_id_mutex_;
  // A small pool used for small objects that last until the
  // Services Manager shuts down. Used by all sub-services.
  std::shared_ptr<MemoryPool> pool_;
  std::vector<std::unique_ptr<Service>> hook_;

  Services();

  Status CreateAllInstances();
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SERVICES_H_
