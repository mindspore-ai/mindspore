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

#include "runtime/device/stream_synchronizer.h"
#include "utils/ms_context.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "include/backend/distributed/recovery/recovery_context.h"

namespace mindspore {
namespace device {
using distributed::collective::CollectiveManager;
using distributed::recovery::RecoveryContext;

std::mutex StreamSynchronizer::instance_lock_;
std::shared_ptr<StreamSynchronizer> StreamSynchronizer::instance_ = nullptr;

void StreamSynchronizer::Initialize() {
  // Non disaster recovery mode does not need to start thread and timeout mechanisms.
  if (!RecoveryContext::GetInstance()->enable_recovery()) {
    return;
  }

  worker_thread_ = std::thread(&StreamSynchronizer::DoSyncStreamTask, this);
}

void StreamSynchronizer::Finalize() {
  {
    std::unique_lock<std::mutex> lock(task_mutex_);
    stop_ = true;
  }
  do_sync_stream_cv_.notify_all();
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  device_context_ = nullptr;
}

bool StreamSynchronizer::SyncStream(const std::string &device_name, uint32_t timeout) {
  std::unique_lock<std::mutex> reentrant_lock(reentrant_mutex_);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  uint32_t device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name, device_id});
  MS_EXCEPTION_IF_NULL(device_context);

  // If disable recovery or timeout==0, sync stream directly to improve performance.
  if (!RecoveryContext::GetInstance()->enable_recovery() || timeout == 0) {
    device_context->Initialize();
    return device_context->device_res_manager_->SyncAllStreams();
  }

  std::unique_lock<std::mutex> lock(task_mutex_);
  if (stop_) {
    MS_LOG(EXCEPTION) << "The synchronization stream task has stopped";
  }
  device_context_ = device_context;
  do_sync_stream_cv_.notify_one();

  if (time_out_cv_.wait_for(lock, std::chrono::seconds(timeout)) == std::cv_status::no_timeout) {
    if (!sync_stream_ret_) {
      MS_LOG(ERROR) << "Synchronize stream failed.";
    }
    return sync_stream_ret_;
  } else {
    CollectiveManager::instance()->set_need_reinit(true);
    if (!CollectiveManager::instance()->Finalize()) {
      MS_LOG(ERROR) << "Finalize collective manager failed.";
      return false;
    }
    time_out_cv_.wait(lock, [this]() { return device_context_ == nullptr; });
    MS_LOG(WARNING) << "Synchronize stream timeout.";
    return true;
  }
}

void StreamSynchronizer::DoSyncStreamTask() {
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(task_mutex_);
      do_sync_stream_cv_.wait(lock, [this]() { return stop_ || device_context_ != nullptr; });
      if (stop_) {
        return;
      }
    }

    device_context_->Initialize();
    // Really sync stream.
    sync_stream_ret_ = device_context_->device_res_manager_->SyncAllStreams();

    {
      std::unique_lock<std::mutex> lock(task_mutex_);
      device_context_ = nullptr;
    }
    time_out_cv_.notify_one();
  }
}
}  // namespace device
}  // namespace mindspore
