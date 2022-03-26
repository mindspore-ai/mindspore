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

#include "plugin/device/gpu/hal/device/gpu_buffer_mgr.h"
#include <cuda_runtime_api.h>
#include <utility>
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace mindspore {
namespace device {
GpuBufferMgr &GpuBufferMgr::GetInstance() noexcept {
  static GpuBufferMgr instance;
  return instance;
}

BlockQueueStatus_T GpuBufferMgr::Create(const std::string &channel_name, void *addr, const std::vector<size_t> &shape,
                                        const size_t &capacity) {
  MS_LOG(INFO) << "Gpu queue: " << channel_name << " created.";
  if (name_queue_map_.count(channel_name)) {
    MS_LOG(ERROR) << "Queue already exist: " << channel_name;
    return QUEUE_EXIST;
  }
  std::shared_ptr<BlockingQueue> queue = std::make_shared<BlockingQueue>();
  BlockQueueStatus_T rt = queue->Create(addr, shape, capacity);
  if (rt != SUCCESS) {
    MS_LOG(ERROR) << "Queue: " << channel_name << "create failed: " << rt;
    return rt;
  }
  (void)name_queue_map_.insert(std::make_pair(channel_name, queue));
  init_ = true;
  return SUCCESS;
}

BlockQueueStatus_T GpuBufferMgr::Open(const std::string &channel_name, const std::vector<size_t> &shape,
                                      const std::function<void(void *, int32_t)> func) {
  MS_LOG(INFO) << "Gpu queue: " << channel_name << " open.";
  set_device();
  if (!name_queue_map_.count(channel_name)) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return QUEUE_NOT_EXIST;
  }

  name_queue_map_[channel_name]->RegisterRelease(func);
  open_by_dataset_++;
  return SUCCESS;
}

BlockQueueStatus_T GpuBufferMgr::Open(const std::string &channel_name, const std::vector<size_t> &shape) {
  MS_LOG(INFO) << "Gpu queue: " << channel_name << " open.";
  set_device();
  if (!name_queue_map_.count(channel_name)) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return QUEUE_NOT_EXIST;
  }
  return SUCCESS;
}

void GpuBufferMgr::set_device_id(int device_id) { cur_dev_id_ = device_id; }

void GpuBufferMgr::set_device() const {
  auto ret = cudaSetDevice(cur_dev_id_);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR)
      << "Set device for id:" << cur_dev_id_ << " failed, ret[" << static_cast<int>(ret) << "], "
      << cudaGetErrorString(ret)
      << ". Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU). "
         "If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be the number set "
         "in the environment variable 'CUDA_VISIBLE_DEVICES'. For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the "
         "'device_id' can be 0,1,2 at the moment, 'device_id' starts from 0, and 'device_id'=0 means using GPU of "
         "number 4.";
  }
}

BlockQueueStatus_T GpuBufferMgr::Push(const std::string &channel_name, const std::vector<DataItemGpu> &data,
                                      unsigned int timeout_in_sec) {
  auto iter = name_queue_map_.find(channel_name);
  if (iter == name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return QUEUE_NOT_EXIST;
  }
  return iter->second->Push(data, timeout_in_sec);
}

BlockQueueStatus_T GpuBufferMgr::Front(const std::string &channel_name, std::vector<DataItemGpu> *data) {
  auto iter = name_queue_map_.find(channel_name);
  if (iter == name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return QUEUE_NOT_EXIST;
  }

  return iter->second->Front(data);
}

BlockQueueStatus_T GpuBufferMgr::Pop(const std::string &channel_name) {
  auto iter = name_queue_map_.find(channel_name);
  if (iter == name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return QUEUE_NOT_EXIST;
  }

  return iter->second->Pop();
}

BlockQueueStatus_T GpuBufferMgr::Clear(const std::string &channel_name) {
  auto iter = name_queue_map_.find(channel_name);
  if (iter == name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return QUEUE_NOT_EXIST;
  }

  return iter->second->Clear();
}

void GpuBufferMgr::Close(const std::string &channel_name) noexcept {
  MS_LOG(INFO) << "Close the queue: " << channel_name;
  return;
}

bool GpuBufferMgr::IsInit() const { return init_; }

bool GpuBufferMgr::IsClosed() const { return closed_; }

bool GpuBufferMgr::Destroy() {
  MS_LOG(INFO) << "Destroy all GPU queue.";
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

inline bool GpuBufferMgr::isCreated(const std::string &channel_name) {
  if (name_queue_map_.count(channel_name) != 0) {
    return true;
  }
  return false;
}

bool GpuBufferMgr::CloseNotify() {
  py::gil_scoped_release release;
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

size_t GpuBufferMgr::Size(const std::string &channel_name) {
  if (!name_queue_map_.count(channel_name)) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return 0;
  }
  return name_queue_map_.at(channel_name)->Size();
}

size_t GpuBufferMgr::Capacity(const std::string &channel_name) {
  if (!name_queue_map_.count(channel_name)) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return 0;
  }
  return name_queue_map_.at(channel_name)->Capacity();
}
}  // namespace device
}  // namespace mindspore
