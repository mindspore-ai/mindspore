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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_SYNC_STREAM_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_SYNC_STREAM_H_

#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "runtime/hardware/device_context.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
constexpr uint32_t kTimeoutInSeconds = 30;

// Execute synchronization stream with timeout mechanism. Typical application scenarios: it is used to monitor
// distributed data parallel training scenarios and whether a process exits unexpectedly.
class BACKEND_EXPORT StreamSynchronizer {
 public:
  static std::shared_ptr<StreamSynchronizer> &GetInstance() {
    std::lock_guard<std::mutex> lock(instance_lock_);
    if (instance_ == nullptr) {
      instance_.reset(new (std::nothrow) StreamSynchronizer());
      MS_EXCEPTION_IF_NULL(instance_);
      instance_->Initialize();
    }
    return instance_;
  }

  ~StreamSynchronizer() = default;

  // Execute synchronization stream with timeout mechanism.
  bool SyncStream(const std::string &device_name, uint32_t timeout = kTimeoutInSeconds);

  // Initialize stream synchronizer, Create a thread to actually execute the synchronization stream task.
  void Initialize();

  // Finalize stream synchronizer, wait worker_thread_ finish.
  void Finalize();

 private:
  StreamSynchronizer() = default;

  DISABLE_COPY_AND_ASSIGN(StreamSynchronizer);

  // Monitor whether there are synchronization stream tasks, and actually execute the synchronization stream
  // tasks.
  void DoSyncStreamTask();

  // Used for multi-thread safety of singleton creation.
  static std::mutex instance_lock_;

  // The singleton pointer.
  static std::shared_ptr<StreamSynchronizer> instance_;

  // Return value of synchronization stream.
  bool sync_stream_ret_{false};

  // Whether synchronization stream thread need to stop.
  bool stop_{false};

  DeviceContext *device_context_{nullptr};

  // The method SyncStream is not multiple threads safe, so use this lock to prevent simultaneous access by
  // multiple threads.
  std::mutex reentrant_mutex_;

  // Use this lock to ensure the safety of external calls to SyncStream and the execution of DoSyncStreamTask
  // in worker_thread_;
  std::mutex task_mutex_;

  // The thread to actually execute the synchronization stream task.
  std::thread worker_thread_;

  std::condition_variable time_out_cv_;
  std::condition_variable do_sync_stream_cv_;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_SYNC_STREAM_H_
