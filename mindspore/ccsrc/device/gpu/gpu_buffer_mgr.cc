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

#include "device/gpu/gpu_buffer_mgr.h"
#include <cuda_runtime_api.h>
#include <utility>
#include "utils/log_adapter.h"
#include "common/utils.h"

namespace mindspore {
namespace device {
unsigned int HandleMgr::AllocHandle() {
  for (size_t i = 0; i < MAX_HANDLE_NUM; ++i) {
    if (!handle_list_[i]) {
      handle_list_[i] = true;
      return (unsigned int)i;
    }
  }
  return INVALID_HANDLE;
}

void HandleMgr::FreeHandle(unsigned int handle_id) {
  if (handle_id >= MAX_HANDLE_NUM) {
    return;
  }
  handle_list_[handle_id] = false;
}

GpuBufferMgr &GpuBufferMgr::GetInstance() noexcept {
  static GpuBufferMgr instance;
  return instance;
}

BlockQueueStatus_T GpuBufferMgr::Create(unsigned int device_id, const std::string &channel_name, void *addr,
                                        const std::vector<size_t> &shape, const size_t &capacity) {
  std::string name = std::to_string(device_id) + std::string("_") + channel_name;
  if (name_queue_map_.count(name)) {
    MS_LOG(ERROR) << "Queue not exist " << name;
    return QUEUE_NOT_EXIST;
  }
  std::shared_ptr<BlockingQueue> queue = std::make_shared<BlockingQueue>();
  BlockQueueStatus_T rt = queue->Create(addr, shape, capacity);
  if (rt != SUCCESS) {
    return rt;
  }
  (void)name_queue_map_.insert(std::make_pair(name, queue));
  init_ = true;
  return SUCCESS;
}

unsigned int GpuBufferMgr::Open(unsigned int device_id, const std::string &channel_name,
                                const std::vector<size_t> &shape, const std::function<void(void *)> func) {
  set_device();
  std::string name = std::to_string(device_id) + std::string("_") + channel_name;
  if (!name_queue_map_.count(name)) {
    MS_LOG(ERROR) << "Queue not exist " << name;
    return HandleMgr::INVALID_HANDLE;
  }
  unsigned int handle = handle_mgr_.AllocHandle();
  if (handle == HandleMgr::INVALID_HANDLE) {
    MS_LOG(ERROR) << "handle is invalid";
    return HandleMgr::INVALID_HANDLE;
  }
  (void)handle_queue_map_.insert(std::make_pair(handle, name_queue_map_[name]));
  name_queue_map_[name]->RegisterRelease(func);
  open_by_dataset_++;
  return handle;
}

unsigned int GpuBufferMgr::Open(unsigned int device_id, const std::string &channel_name,
                                const std::vector<size_t> &shape) {
  set_device();
  std::string name = std::to_string(device_id) + std::string("_") + channel_name;
  if (!name_queue_map_.count(name)) {
    MS_LOG(ERROR) << "Queue not exist " << name;
    return HandleMgr::INVALID_HANDLE;
  }
  unsigned int handle = handle_mgr_.AllocHandle();
  if (handle == HandleMgr::INVALID_HANDLE) {
    MS_LOG(ERROR) << "handle is invalid";
    return HandleMgr::INVALID_HANDLE;
  }
  (void)handle_queue_map_.insert(std::make_pair(handle, name_queue_map_[name]));
  return handle;
}

void GpuBufferMgr::set_device_id(int device_id) { cur_dev_id_ = device_id; }

void GpuBufferMgr::set_device() const {
  auto ret = cudaSetDevice(cur_dev_id_);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaSetDevice, ret[" << static_cast<int>(ret) << "]";
  }
}

BlockQueueStatus_T GpuBufferMgr::Push(unsigned int handle, const std::vector<DataItemGpu> &data,
                                      unsigned int timeout_in_sec) {
  auto iter = handle_queue_map_.find(handle);
  if (iter == handle_queue_map_.end()) {
    return HANDLE_NOT_EXIST;
  }
  return iter->second->Push(data, timeout_in_sec);
}

BlockQueueStatus_T GpuBufferMgr::Front(unsigned int handle, void **addr, size_t *len) {
  auto iter = handle_queue_map_.find(handle);
  if (iter == handle_queue_map_.end()) {
    return HANDLE_NOT_EXIST;
  }
  return iter->second->Front(addr, len);
}

BlockQueueStatus_T GpuBufferMgr::Pop(unsigned int handle) {
  auto iter = handle_queue_map_.find(handle);
  if (iter == handle_queue_map_.end()) {
    return HANDLE_NOT_EXIST;
  }
  return iter->second->Pop();
}

void GpuBufferMgr::Close(unsigned int handle) noexcept {
  if (!handle_queue_map_.count(handle)) {
    return;
  }
  (void)handle_queue_map_.erase(handle);
  handle_mgr_.FreeHandle(handle);
  return;
}

bool GpuBufferMgr::IsInit() const { return init_; }

bool GpuBufferMgr::IsClosed() const { return closed_; }

bool GpuBufferMgr::Destroy() {
  for (auto iter = name_queue_map_.begin(); iter != name_queue_map_.end(); ++iter) {
    std::shared_ptr<BlockingQueue> queue = iter->second;
    if (queue != nullptr) {
      if (!queue->Destroy()) {
        return false;
      }
      queue.reset();
    }
  }
  name_queue_map_.clear();
  return true;
}

inline bool GpuBufferMgr::isCreated(unsigned int device_id, const std::string &channel_name) {
  std::string name = std::to_string(device_id) + std::string("_") + channel_name;
  if (name_queue_map_.count(name) != 0) {
    return true;
  }
  return false;
}

bool GpuBufferMgr::CloseNotify() {
  bool result = true;
  // lock scope
  {
    std::lock_guard<std::mutex> lk(close_mutex_);
    // set closed_ to be true, all the dataset retry can be jumped out of the while
    closed_ = true;
  }

  // wati for the dataset threads' ack
  for (int i = 0; i < open_by_dataset_; i++) {
    if (sema.Wait() == false) {
      MS_LOG(ERROR) << "time out of receiving signals";
      result = false;
    }
    MS_LOG(DEBUG) << "receive one signal (" << i + 1 << "/" << open_by_dataset_ << ")";
  }
  return result;
}

void GpuBufferMgr::CloseConfirm() { sema.Signal(); }
}  // namespace device
}  // namespace mindspore
