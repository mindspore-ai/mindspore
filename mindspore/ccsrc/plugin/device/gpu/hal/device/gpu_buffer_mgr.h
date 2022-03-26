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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_BUFFER_MGR_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_BUFFER_MGR_H_

#include <unistd.h>
#include <cstring>
#include <iostream>
#include <functional>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "plugin/device/gpu/hal/device/blocking_queue.h"

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

class GpuBufferMgr {
 public:
  EXPORT GpuBufferMgr() : cur_dev_id_(0), init_(false), closed_(false), open_by_dataset_(0) {}

  EXPORT virtual ~GpuBufferMgr() = default;

  EXPORT static GpuBufferMgr &GetInstance() noexcept;

  EXPORT BlockQueueStatus_T Create(const std::string &channel_name, void *addr, const std::vector<size_t> &shape,
                                   const size_t &capacity);

  // call for Push thread
  EXPORT BlockQueueStatus_T Open(const std::string &channel_name, const std::vector<size_t> &shape,
                                 std::function<void(void *, int32_t)> func);

  // call for Front/Pop thread
  EXPORT BlockQueueStatus_T Open(const std::string &channel_name, const std::vector<size_t> &shape);

  EXPORT BlockQueueStatus_T Push(const std::string &channel_name, const std::vector<DataItemGpu> &data,
                                 unsigned int timeout_in_sec);
  EXPORT BlockQueueStatus_T Front(const std::string &channel_name, std::vector<DataItemGpu> *data);
  EXPORT BlockQueueStatus_T Pop(const std::string &channel_name);
  EXPORT BlockQueueStatus_T Clear(const std::string &channel_name);

  EXPORT void set_device_id(int device_id);

  EXPORT void Close(const std::string &channel_name) noexcept;

  EXPORT bool IsInit() const;

  EXPORT bool IsClosed() const;

  EXPORT bool Destroy();

  // call for Release GPU Resources
  EXPORT bool CloseNotify();

  // call for dataset send thread
  EXPORT void CloseConfirm();

  EXPORT size_t Size(const std::string &channel_name);

  EXPORT size_t Capacity(const std::string &channel_name);

 private:
  void set_device() const;

  int cur_dev_id_;
  bool init_;
  bool closed_;
  std::mutex mutex_;
  std::mutex close_mutex_;
  // how many queues opened by dataset
  int open_by_dataset_;
  Semaphore sema;

  std::map<std::string, std::shared_ptr<BlockingQueue>> name_queue_map_;

  inline bool isCreated(const std::string &channel_name);

  GpuBufferMgr(const GpuBufferMgr &) = delete;
  GpuBufferMgr &operator=(const GpuBufferMgr &) = delete;
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_BUFFER_MGR_H_
