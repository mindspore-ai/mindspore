/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "runtime/device/multi_stream_controller.h"

#include <algorithm>

namespace mindspore {
namespace device {
constexpr size_t kDefaultStreamRefreshSize = 2;

MultiStreamControllerPtr &MultiStreamController::GetInstance() {
  static std::once_flag init_flag = {};
  static MultiStreamControllerPtr multi_stream_controller = nullptr;
  std::call_once(init_flag, [&]() {
    if (multi_stream_controller == nullptr) {
      MS_LOG(INFO) << "Create MultiStreamController.";
      multi_stream_controller = std::make_shared<MultiStreamController>();
    }
  });

  return multi_stream_controller;
}

void MultiStreamController::Refresh(const DeviceContext *device_context) {
  auto stream_size = device_context->device_res_manager_->QueryStreamSize();
  MS_LOG(INFO) << "Stream manager initialize, device_context : " << device_context << ", stream_size : " << stream_size
               << ".";
  if (stream_size == 0) {
    // CPU has no concept of stream, stream size must be zero.
    MS_LOG(INFO) << "Stream size is 0, will initialize with 2 streams.";
    stream_size = kDefaultStreamRefreshSize;
  }
  task_id_on_stream_manager_[device_context].Resize(stream_size);
  if (event_pools_.count(device_context) == 0) {
    (void)event_pools_.emplace(device_context, std::make_shared<EventPool>([device_context]() {
                                 // Event in pool need to do synchronization between streams, need to enable blocking.
                                 return device_context->device_res_manager_->CreateRuntimeEvent(true, false);
                               }));
  }
}

bool MultiStreamController::UpdateTaskIdOnStream(const DeviceContext *device_context, int64_t task_id_on_stream,
                                                 uint32_t user_stream_id, uint32_t memory_stream_id) {
  return task_id_on_stream_manager_[device_context].Update(task_id_on_stream, user_stream_id, memory_stream_id);
}

int64_t MultiStreamController::QueryTaskIdOnStream(const DeviceContext *device_context, uint32_t user_stream_id,
                                                   uint32_t memory_stream_id) {
  return task_id_on_stream_manager_[device_context].Query(user_stream_id, memory_stream_id);
}

int64_t MultiStreamController::LaunchTaskIdOnStream(const DeviceContext *device_context, uint32_t stream_id) {
  auto iter = task_id_on_stream_manager_.find(device_context);
  if (iter == task_id_on_stream_manager_.end()) {
    if (device_context->GetDeviceType() == DeviceType::kCPU) {
      return INT64_MAX;
    }

    MS_LOG(WARNING) << "LaunchTaskIdOnStream device context is not found, device_context name : "
                    << device_context->device_context_key().device_name_ << ", stream id : " << stream_id
                    << ", refresh context.";
    Refresh(device_context);
    return task_id_on_stream_manager_[device_context].Launch(stream_id);
  }
  return iter->second.Launch(stream_id);
}

int64_t MultiStreamController::GetTaskIdOnStream(const DeviceContext *device_context, uint32_t stream_id) {
  return task_id_on_stream_manager_[device_context].Get(stream_id);
}

std::mutex &MultiStreamController::GetStreamMutex(const DeviceContext *device_context, size_t stream_id) {
  return stream_mutexes_[device_context][stream_id];
}

bool MultiStreamController::RecordEvent(const DeviceContext *device_context, int64_t task_id_on_stream,
                                        uint32_t user_stream_id,
                                        const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses) {
  auto mem_manager = device_context->device_res_manager_->mem_manager();
  if (mem_manager == nullptr) {
    MS_LOG(WARNING) << "mem_manager_ is nullptr.";
    return false;
  }

  auto event = device_context->device_res_manager_->CreateRuntimeEvent(false, true);
  if (event == nullptr) {
    return true;
  }
  event->RecordEvent(user_stream_id);
  // Record event on mem buf.
  return mem_manager->RecordEvent(task_id_on_stream, user_stream_id, memory_stream_addresses, event);
}

bool MultiStreamController::WaitEvent(const DeviceContext *device_context, int64_t task_id_on_stream,
                                      uint32_t user_stream_id, uint32_t memory_stream_id) {
  auto mem_manager = device_context->device_res_manager_->mem_manager();
  if (mem_manager == nullptr) {
    MS_LOG(WARNING) << "mem_manager_ is nullptr.";
    return false;
  }
  // If update task id on stream failed, means task id on stream is elder one, no need to wait event on mem manager.
  if (!UpdateTaskIdOnStream(device_context, task_id_on_stream, user_stream_id, memory_stream_id)) {
    MS_LOG(DEBUG) << "Skip Wait Event.";
    return false;
  }
  return mem_manager->WaitEvent(task_id_on_stream, user_stream_id, memory_stream_id);
}

bool MultiStreamController::WaitEvent(const DeviceContext *device_context, int64_t task_id_on_stream,
                                      uint32_t user_stream_id) {
  auto mem_manager = device_context->device_res_manager_->mem_manager();
  if (mem_manager == nullptr) {
    MS_LOG(WARNING) << "mem_manager_ is nullptr.";
    return false;
  }

  return mem_manager->WaitEvent(task_id_on_stream, user_stream_id);
}

bool MultiStreamController::DispatchRecordWaitEvent(const DeviceContext *device_context, uint32_t user_stream_id,
                                                    uint32_t memory_stream_id) {
  if (event_pools_.count(device_context) == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "device context has not initialized.";
  }
  auto &event_pool = event_pools_[device_context];
  auto event = event_pool->Get();
  // Note : record event on memory stream id and wait event on user stream id to make sure memory is safe.
  event->RecordEvent(memory_stream_id);
  event->WaitEvent(user_stream_id);
  return true;
}

bool MultiStreamController::SyncStream(const DeviceContext *device_context, size_t stream_id) {
  auto &device_res_manager = device_context->device_res_manager_;
  bool ret = device_res_manager->SyncStream(stream_id);
  auto mem_manager = device_res_manager->mem_manager();
  if (mem_manager != nullptr) {
    auto task_id_on_stream = GetTaskIdOnStream(device_context, stream_id);
    mem_manager->WaitEvent(task_id_on_stream, stream_id);
  }
  return ret;
}

bool MultiStreamController::SyncAllStreams(const DeviceContext *device_context) {
  auto &device_res_manager = device_context->device_res_manager_;
  bool ret = device_res_manager->SyncAllStreams();
  auto mem_manager = device_res_manager->mem_manager();
  if (mem_manager != nullptr) {
    mem_manager->SyncAllEvents();
  }
  return ret;
}

bool MultiStreamController::SyncNotDefaultStreams(const DeviceContext *device_context) {
  auto &device_res_manager = device_context->device_res_manager_;
  bool ret = device_res_manager->SyncNotDefaultStreams();
  auto mem_manager = device_res_manager->mem_manager();
  if (mem_manager != nullptr) {
    auto stream_ids = device_res_manager->GetStreamIds();
    for (auto stream_id : stream_ids) {
      auto task_id_on_stream = GetTaskIdOnStream(device_context, stream_id);
      mem_manager->WaitEvent(task_id_on_stream, stream_id);
    }
  }
  return ret;
}

void TaskIdOnStreamManager::Resize(uint32_t stream_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_ && stream_size <= initialize_size_) {
    MS_LOG(INFO) << "Task id on stream manager has already initialized, current size : " << initialize_size_ << ".";
    return;
  }
  MS_LOG(INFO) << "Task id on stream manager initialize : " << initialized_ << ", stream_size : " << stream_size << ".";
  uint32_t min_stream_size = 2;
  initialize_size_ = std::max(stream_size, min_stream_size);
  generator_.resize(initialize_size_);
  status_.resize(initialize_size_);
  for (auto &vec : status_) {
    vec.resize(initialize_size_);
  }
  initialized_ = true;
}

int64_t TaskIdOnStreamManager::Query(uint32_t user_stream_id, uint32_t memory_stream_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return status_[user_stream_id][memory_stream_id];
}

bool TaskIdOnStreamManager::Update(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (status_[user_stream_id][memory_stream_id] >= task_id_on_stream) {
    return false;
  }
  status_[user_stream_id][memory_stream_id] = task_id_on_stream;
  return true;
}

int64_t TaskIdOnStreamManager::Launch(uint32_t stream_id) {
  if (stream_id >= generator_.size()) {
    MS_LOG(WARNING) << "Launch stream id : " << stream_id << " failed, generator_ size : " << generator_.size();
    generator_.resize(stream_id + 1);
    status_.resize(stream_id + 1);
  }
  return ++generator_[stream_id].value_;
}

int64_t TaskIdOnStreamManager::Get(uint32_t stream_id) { return generator_[stream_id].value_; }

EventPool::EventPool(std::function<DeviceEventPtr(void)> event_creator) : event_creator_(std::move(event_creator)) {}

EventPool::~EventPool() {
  std::lock_guard<std::mutex> lock(mutex_);
  expired_ = true;
  events_.clear();
  cached_events_.clear();
}

DeviceEventPtr EventPool::Get() {
  MS_LOG(DEBUG) << "Event pool get start.";
  std::lock_guard<std::mutex> lock(mutex_);
  DeviceEvent *event = nullptr;
  // Try to create event firstly before reached core size.
  if (size_ < core_size_) {
    auto created_event = event_creator_();
    if (created_event != nullptr && created_event->IsReady()) {
      cached_events_.push_back(created_event);
      size_++;
      event = created_event.get();
    }
  }
  // Try to reuse event.
  if (event == nullptr) {
    auto iter = events_.begin();
    while (iter != events_.end()) {
      auto event_in_list = *iter;
      if (event_in_list == nullptr) {
        MS_LOG(INTERNAL_EXCEPTION) << "exception : event in list is nullptr, events_ size : " << events_.size() << ".";
      }
      if (event_in_list->QueryEvent()) {
        event = event_in_list;
        events_.erase(iter);
        break;
      }
      iter++;
    }
  }
  // Reuse failed, try to create more event.
  if (event == nullptr) {
    auto created_event = event_creator_();
    if (created_event != nullptr && created_event->IsReady()) {
      cached_events_.push_back(created_event);
      event = created_event.get();
      size_++;
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Get event failed.";
    }
  }
  MS_LOG(DEBUG) << "Get event, events_ size : " << events_.size() << ", event : " << event << ".";

  auto event_ptr = std::shared_ptr<DeviceEvent>(event, [&](DeviceEvent *e) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!expired_) {
      MS_LOG(DEBUG) << "Return event : " << e << ".";
      events_.push_back(e);
    } else {
      MS_LOG(DEBUG) << "Return event : " << e << "failed.";
    }
  });
  return event_ptr;
}
}  // namespace device
}  // namespace mindspore
