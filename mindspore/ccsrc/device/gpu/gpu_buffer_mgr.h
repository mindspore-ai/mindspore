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

#ifndef MINDSPORE_CCSRC_DEVICE_GPU_GPU_BUFFER_MGR_H_
#define MINDSPORE_CCSRC_DEVICE_GPU_GPU_BUFFER_MGR_H_

#include <unistd.h>
#include <cstring>
#include <iostream>
#include <functional>
#include <map>
#include <string>
#include <memory>
#include "device/gpu/blocking_queue.h"

#define EXPORT __attribute__((visibility("default")))

namespace mindspore {
namespace device {
static const unsigned int MAX_WAIT_TIME_IN_SEC = 60;

class Semaphore {
 public:
  explicit Semaphore(int count = 0) : count_(count) {}

  inline void Signal() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++count_;
    cv_.notify_one();
  }

  inline bool Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (count_ == 0) {
      if (cv_.wait_for(lock, std::chrono::seconds(MAX_WAIT_TIME_IN_SEC)) == std::cv_status::timeout) {
        return false;
      }
    }
    --count_;
    return true;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int count_;
};

class HandleMgr {
 public:
  static const unsigned int MAX_HANDLE_NUM = 32;
  static const unsigned int INVALID_HANDLE = 0xffffffffUL;

  unsigned int AllocHandle();
  void FreeHandle(unsigned int);

 private:
  bool handle_list_[MAX_HANDLE_NUM];
};

class GpuBufferMgr {
 public:
  EXPORT GpuBufferMgr() : cur_dev_id_(0), init_(false), closed_(false), open_by_dataset_(0) {}

  EXPORT virtual ~GpuBufferMgr() = default;

  EXPORT static GpuBufferMgr &GetInstance() noexcept;

  EXPORT BlockQueueStatus_T Create(unsigned int device_id, const std::string &channel_name, void *addr,
                                   const size_t &feature_len, const size_t &label_size, const size_t &capacity);

  // call for Push thread
  EXPORT unsigned int Open(unsigned int device_id, const std::string &channel_name, const size_t &feature_len,
                           const size_t &label_size, std::function<void(void *)> func);

  // call for Front/Pop thread
  EXPORT unsigned int Open(unsigned int device_id, const std::string &channel_name, const size_t &feature_len,
                           const size_t &label_size);

  EXPORT BlockQueueStatus_T Push(unsigned int handle, void *feature_addr, size_t feature_size, void *label_addr,
                                 size_t label_size, unsigned int timeout_in_sec);
  EXPORT BlockQueueStatus_T Front(unsigned int handle, void **feature_addr, size_t *feature_size, void **label_addr,
                                  size_t *label_size);
  EXPORT BlockQueueStatus_T Pop(unsigned int handle);

  EXPORT void set_device_id(int device_id);

  EXPORT void Close(unsigned int handle) noexcept;

  EXPORT bool IsInit() const;

  EXPORT bool IsClosed() const;

  EXPORT bool Destroy();

  // call for Release GPU Resources
  EXPORT bool CloseNotify();

  // call for dataset send thread
  EXPORT void CloseConfirm();

 private:
  void set_device() const;

  int cur_dev_id_;
  bool init_;
  bool closed_;
  std::mutex mutex_;
  std::mutex close_mutex_;
  std::condition_variable close_confirm_cond_;
  // how many queues opened by dataset
  int open_by_dataset_;
  Semaphore sema;

  HandleMgr handle_mgr_;

  std::map<unsigned int, std::shared_ptr<BlockingQueue>> handle_queue_map_;
  std::map<std::string, std::shared_ptr<BlockingQueue>> name_queue_map_;

  inline bool isCreated(unsigned int device_id, const std::string &channel_name);

  GpuBufferMgr(const GpuBufferMgr &) = delete;
  GpuBufferMgr &operator=(const GpuBufferMgr &) = delete;
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_GPU_GPU_BUFFER_MGR_H_
