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

#include "include/backend/data_queue/data_queue_mgr.h"
#include <algorithm>
#include <utility>
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "utils/ms_context.h"
#include "pybind11/pybind11.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/data_queue/blocking_queue.h"
#include "runtime/device/kernel_info.h"

namespace py = pybind11;

namespace mindspore {
namespace device {
DataQueueMgr &DataQueueMgr::GetInstance() noexcept {
  static DataQueueMgr instance;
  return instance;
}

void DataQueueMgr::RegisterDataQueueCreator(const std::string &device_name, DataQueueCreator &&creator) {
  data_queue_creator_map_.emplace(device_name, std::forward<DataQueueCreator>(creator));
}

std::shared_ptr<DataQueue> DataQueueMgr::CreateDataQueue(const std::string &device_name,
                                                         const std::string &channel_name, bool dynamic_shape,
                                                         size_t capacity, void *addr,
                                                         const std::vector<size_t> &shape) {
  auto iter = data_queue_creator_map_.find(device_name);
  if (iter == data_queue_creator_map_.end()) {
    return nullptr;
  }

  return iter->second(channel_name, dynamic_shape, capacity, addr, shape);
}

DataQueueStatus DataQueueMgr::Create(const std::string &channel_name, void *addr, const std::vector<size_t> &shape,
                                     const size_t &capacity) {
  MS_LOG(INFO) << "Static GPU queue: " << channel_name << " created";
  if (name_queue_map_.find(channel_name) != name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue already exist: " << channel_name;
    return DataQueueStatus::QUEUE_EXIST;
  }

  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  std::shared_ptr<DataQueue> data_queue = CreateDataQueue(
    MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET), channel_name, false, capacity, addr, shape);
  if (data_queue != nullptr) {
    std::shared_ptr<BlockingQueue> queue = std::make_shared<BlockingQueue>();
    DataQueueStatus rt = queue->Create(data_queue);
    if (rt != DataQueueStatus::SUCCESS) {
      MS_LOG(ERROR) << "Queue: " << channel_name << "create failed: " << rt;
      return rt;
    }
    (void)name_queue_map_.insert(std::make_pair(channel_name, queue));
    init_ = true;
    return DataQueueStatus::SUCCESS;
  }

  MS_LOG(ERROR) << "Static data queue only support GPU target.";
  return DataQueueStatus::INTERNAL_ERROR;
}

DataQueueStatus DataQueueMgr::Open(const std::string &channel_name, const std::function<void(void *, int32_t)> func) {
  MS_LOG(INFO) << "Gpu queue: " << channel_name << " open.";
  if (name_queue_map_.find(channel_name) == name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return DataQueueStatus::QUEUE_NOT_EXIST;
  }

  name_queue_map_[channel_name]->RegisterRelease(func);
  open_by_dataset_++;
  return DataQueueStatus::SUCCESS;
}
DataQueueStatus DataQueueMgr::OpenDynamicBufQueue(const std::string &channel_name,
                                                  const std::function<void(void *, int32_t)> func) {
  std::unique_lock<std::mutex> locker(mutex_);
  if (name_queue_map_.find(channel_name) == name_queue_map_.end()) {
    DataQueueStatus status = CreateDynamicBufQueue(channel_name, default_capacity_);
    MS_EXCEPTION_IF_CHECK_FAIL(status == DataQueueStatus::SUCCESS, "Create dynamic buffer queue failed");
    MS_LOG_INFO << "Create dynamic buffer queue: " << channel_name;
    cv_.notify_all();
  }
  name_queue_map_[channel_name]->RegisterRelease(func);
  open_by_dataset_++;
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus DataQueueMgr::OpenDynamicBufQueue(const std::string &channel_name) {
  std::unique_lock<std::mutex> locker(mutex_);
  auto time_out = cv_.wait_for(locker, std::chrono::seconds(MAX_WAIT_TIME_IN_SEC),
                               [this, &channel_name] { return name_queue_map_.count(channel_name); });
  if (!time_out) {
    return DataQueueStatus::TIMEOUT;
  }
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus DataQueueMgr::CreateDynamicBufQueue(const std::string &channel_name, const size_t &capacity) {
  if (name_queue_map_.find(channel_name) != name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue already exist: " << channel_name;
    return DataQueueStatus::QUEUE_EXIST;
  }
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  std::string device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  std::shared_ptr<DataQueue> device_queue = CreateDataQueue(device_name, channel_name, true, capacity, nullptr, {});
  if (device_queue != nullptr && (device_name == kGPUDevice || device_name == kAscendDevice)) {
    std::shared_ptr<BlockingQueue> queue = std::make_shared<BlockingQueue>();
    DataQueueStatus rt = queue->Create(device_queue);
    if (rt != DataQueueStatus::SUCCESS) {
      MS_LOG(ERROR) << "Queue: " << channel_name << "create failed: " << rt;
      return rt;
    }
    (void)name_queue_map_.insert(std::make_pair(channel_name, queue));
    init_ = true;
    return DataQueueStatus::SUCCESS;
  }

  MS_LOG(ERROR) << "Dynamic data queue only support Ascend/GPU target, bug got " << device_name;
  return DataQueueStatus::QUEUE_EXIST;
}

DataQueueStatus DataQueueMgr::Open(const std::string &channel_name) const {
  if (name_queue_map_.find(channel_name) == name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return DataQueueStatus::QUEUE_NOT_EXIST;
  }
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus DataQueueMgr::Push(const std::string &channel_name, const std::vector<DataQueueItem> &data,
                                   unsigned int timeout_in_sec) {
  auto iter = name_queue_map_.find(channel_name);
  if (iter == name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return DataQueueStatus::QUEUE_NOT_EXIST;
  }
  return iter->second->Push(data, timeout_in_sec);
}

DataQueueStatus DataQueueMgr::Front(const std::string &channel_name, std::vector<DataQueueItem> *data) {
  auto iter = name_queue_map_.find(channel_name);
  if (iter == name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return DataQueueStatus::QUEUE_NOT_EXIST;
  }

  return iter->second->Front(data);
}

DataQueueStatus DataQueueMgr::Pop(const std::string &channel_name) {
  auto iter = name_queue_map_.find(channel_name);
  if (iter == name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return DataQueueStatus::QUEUE_NOT_EXIST;
  }

  return iter->second->Pop();
}

DataQueueStatus DataQueueMgr::Clear(const std::string &channel_name) {
  auto iter = name_queue_map_.find(channel_name);
  if (iter == name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return DataQueueStatus::QUEUE_NOT_EXIST;
  }

  return iter->second->Clear();
}

void DataQueueMgr::Close(const std::string &channel_name) const noexcept {
  MS_LOG(INFO) << "Close the queue: " << channel_name;
  return;
}

bool DataQueueMgr::IsInit() const { return init_; }

bool DataQueueMgr::IsClosed() const { return closed_; }

bool DataQueueMgr::Destroy() {
  MS_LOG(INFO) << "Destroy all data queue.";
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

inline bool DataQueueMgr::isCreated(const std::string &channel_name) const {
  return name_queue_map_.find(channel_name) != name_queue_map_.end();
}

bool DataQueueMgr::CloseNotify() {
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
    MS_LOG(DEBUG) << "receive one signal (" << (i + 1) << "/" << open_by_dataset_ << ")";
  }
  return result;
}

void DataQueueMgr::CloseConfirm() { sema.Signal(); }

size_t DataQueueMgr::Size(const std::string &channel_name) {
  if (name_queue_map_.find(channel_name) == name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return 0;
  }
  return name_queue_map_.at(channel_name)->Size();
}

size_t DataQueueMgr::Capacity(const std::string &channel_name) {
  if (name_queue_map_.find(channel_name) == name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return 0;
  }
  return name_queue_map_.at(channel_name)->Capacity();
}

std::shared_ptr<DataQueue> DataQueueMgr::GetDataQueue(const std::string &channel_name) const {
  auto iter = name_queue_map_.find(channel_name);
  if (iter == name_queue_map_.end()) {
    MS_LOG(ERROR) << "Queue not exist " << channel_name;
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(iter->second);
  return iter->second->Queue();
}

DataQueueStatus DataQueueMgr::SetThreadDevice(const std::string &device_name) {
  auto tmp_queue = CreateDataQueue(device_name, {}, false, 0, nullptr, {});
  if (tmp_queue == nullptr) {
    return DataQueueStatus::QUEUE_NOT_EXIST;
  }
  tmp_queue->SetThreadDevice();
  return DataQueueStatus::SUCCESS;
}

std::shared_ptr<void> DataQueueMgr::AllocHostMem(const std::string &device_name, size_t size) {
  auto tmp_queue = CreateDataQueue(device_name, {}, false, 0, nullptr, {});
  if (tmp_queue == nullptr) {
    return std::shared_ptr<void>(::malloc(size), ::free);
  }

  return tmp_queue->AllocHostMem(size);
}
#ifndef BUILD_LITE
bool PopDataFromDataQueue(const AnfNodePtr &data_kernel) {
  auto queue_name = common::AnfAlgo::GetNodeAttr<std::string>(data_kernel, "shared_name");
  device::DataQueueMgr &buf_mgr = device::DataQueueMgr::GetInstance();
  auto ret = buf_mgr.OpenDynamicBufQueue(queue_name);
  MS_EXCEPTION_IF_CHECK_FAIL(ret == device::DataQueueStatus::SUCCESS, "Open dynamic data queue failed");
  std::vector<device::DataQueueItem> data;
  auto kernel_info = dynamic_cast<device::KernelInfo *>(data_kernel->kernel_info());
  (void)buf_mgr.Front(queue_name, &data);
  (void)buf_mgr.Pop(queue_name);
  std::vector<std::shared_ptr<device::DeviceAddress>> device_tensors;
  for (auto &device_tensor : kernel_info->output_address_list()) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    device_tensors.push_back(device_tensor);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(data.size() == device_tensors.size(),
                             "The number of data tensor popped from dynamic queue is not correct");
  std::vector<ShapeVector> shapes;
  std::vector<TypeId> types;
  std::vector<size_t> output_size_list;
  for (size_t i = 0; i < data.size(); ++i) {
    device_tensors[i]->set_ptr(data[i].device_addr_);
    device_tensors[i]->SetSize(data[i].data_len_);
    device_tensors[i]->set_from_mem_pool(true);
    output_size_list.push_back(data[i].data_len_);
    (void)kernel_info->SetOutputAddr(device_tensors[i], i);
    shapes.push_back(data[i].shapes_);
    types.push_back(common::AnfAlgo::GetOutputInferDataType(data_kernel, i));
  }
  auto kernel_mod = kernel_info->MutableKernelMod();
  kernel_mod->SetOutputSizeList(output_size_list);
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, data_kernel.get());
  return true;
}
#endif
}  // namespace device
}  // namespace mindspore
