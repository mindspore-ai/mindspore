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

#include "include/pynative/utils/pyboost/comm_handle.h"
#include <memory>
#include "include/pynative/utils/pyboost/comm_utils.h"
#include "include/pynative/utils/runtime/op_executor.h"
#include "include/pynative/utils/runtime/task/device_task.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "utils/current_state.h"
#include "utils/ms_exception.h"
namespace mindspore {
namespace kernel {
namespace pyboost {
class DelGuard {
 public:
  explicit DelGuard(CommHandlePtr comm_handle) : comm_handle_(comm_handle) {}
  ~DelGuard() { comm_handle_->SetWaitCalled(); }

 private:
  CommHandlePtr comm_handle_;
};

CommHandle::~CommHandle() {
  if (event_ == nullptr) {
    return;
  }
  bool wait_called = wait_called_.load();
  if (!wait_called && !op_name_.empty()) {
    MS_LOG(WARNING) << "CommHandle destroyed without calling wait(): event=" << event_ << ", op_name=" << op_name_
                    << ", CommHandle=" << this
                    << ". Please ensure to call wait() on the CommHandle returned from async communication operations.";
  }
  MS_LOG(DEBUG) << "DestroyEvent: event=" << event_ << ", op_name=" << op_name_ << ", wait_called=" << wait_called
                << ", CommHandle=" << this;
  // Don't access device_ctx_ here - it may be a dangling pointer during process exit after ClearResAtexit()
  // has destroyed the DeviceContext. Call DestroyEvent directly on the event object instead.
  try {
    if (!event_->DestroyEvent()) {
      MS_LOG(WARNING) << "DestroyEvent failed in CommHandle destructor, event=" << event_ << ", op_name=" << op_name_;
    }
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "DestroyEvent exception in CommHandle destructor: " << e.what() << ", event=" << event_
                    << ", op_name=" << op_name_;
  }
  MS_LOG(DEBUG) << "DestroyEvent done, event: " << event_;
  event_ = nullptr;
}

DeviceEventPtr CommHandle::CreateEvent() {
  MS_EXCEPTION_IF_NULL(device_ctx_);
  device_ctx_->device_res_manager_->BindDeviceToCurrentThread(false);
  // Runtime events are not retained by DeviceResManager. CommHandle can therefore destroy the event without
  // accessing device_ctx_, which may already be invalid during process exit.
  event_ = device_ctx_->device_res_manager_->CreateRuntimeEvent(true, false);
  if (event_ != nullptr) {
    MS_LOG(DEBUG) << "CreateEvent: event=" << event_ << ", op_name=" << op_name_ << ", CommHandle=" << this;
  }
  return event_;
}

void CommHandle::RecordEvent(size_t stream_id) {
  // Call this function in device thread
  event_->RecordEvent(stream_id);
}

void CommHandle::UpdateTaskId(size_t stream_id) {
  auto &controller = device::DeviceContextManager::GetInstance().GetMultiStreamController(
    device_ctx_->device_context_key().device_type_);
  controller->Refresh();
  auto task_id = controller->LaunchTaskIdOnStream(stream_id);
  *task_id_on_stream_ = task_id;
  *record_stream_id_ = stream_id;
}

void CommHandle::AddMemoryStreamId(size_t stream_id) {
  memory_stream_id_set_.insert(stream_id);
  MS_LOG(DEBUG) << "AddMemoryStreamId: stream_id=" << stream_id
                << ", memory_stream_id_set_size=" << memory_stream_id_set_.size();
}

void CommHandle::GetMemoryStreamIdInfoFromInput(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_LOG(DEBUG) << "GetMemoryStreamIdInfoFromInput: called with tensor=" << tensor.get();
  if (tensor->device_address() == nullptr) {
    MS_LOG(DEBUG) << "GetMemoryStreamIdInfoFromInput: tensor device_address is nullptr, skip";
    return;
  }
  auto device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);
  auto stream_id = device_address->stream_id();
  MS_LOG(DEBUG) << "GetMemoryStreamIdInfoFromInput: extracted stream_id=" << stream_id
                << " from device_address=" << device_address.get();
  AddMemoryStreamId(stream_id);
}

void CommHandle::WaitDeviceEvent(size_t cur_stream_id) {
  MS_EXCEPTION_IF_NULL(device_ctx_);
  if (event_ == nullptr) {
    return;
  }

  auto cur_stream_ptr = device_ctx_->device_res_manager_->GetStream(cur_stream_id);
  event_->set_wait_stream(cur_stream_ptr);
  event_->WaitEventWithoutReset();
}

void CommHandle::ReleaseMultiStreamEvent(size_t cur_stream_id) {
  MS_EXCEPTION_IF_NULL(device_ctx_);
  if (event_ == nullptr) {
    return;
  }

  MS_LOG(DEBUG) << "WaitEvent wait stream id:" << cur_stream_id << ", record_stream_id:" << *record_stream_id_
                << ", event:" << event_ << ", task_id_on_stream:" << *task_id_on_stream_;
  // Release cross stream memory event, mark record_stream_id is use stream id, wait stream id is memory stream
  // id.
  (void)device::DeviceContextManager::GetInstance()
    .GetMultiStreamController(device_ctx_->device_context_key().device_type_)
    ->WaitEvent(*task_id_on_stream_, *record_stream_id_, cur_stream_id);
}

void WaitTaskFunc(CommHandlePtr comm_handle) {
  MS_EXCEPTION_IF_NULL(comm_handle);
  WaitTaskFunc(comm_handle, CurrentStream::id());
}

void WaitTaskFunc(CommHandlePtr comm_handle, size_t cur_stream_id) {
  MS_EXCEPTION_IF_NULL(comm_handle);
  auto wait_fn = [cur_stream_id, comm_handle, del_guard = DelGuard(comm_handle)]() {
    runtime::OpExecutor::DispatchLaunchTask(
      // wait device event, wait_stream_id is the stream id of the device event
      [cur_stream_id, comm_handle]() { comm_handle->WaitDeviceEvent(cur_stream_id); });

    for (auto memory_stream_id : comm_handle->memory_stream_id_set()) {
      comm_handle->ReleaseMultiStreamEvent(memory_stream_id);
      MS_LOG(DEBUG) << "ReleaseMultiStreamEvent: stream_id=" << memory_stream_id;
    }
    comm_handle->SetWaitCalled();
    MS_LOG(DEBUG) << "Finish WaitTaskFunc: comm_handle=" << comm_handle;
  };
  if (!runtime::OpExecutor::NeedSync()) {
    runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
      std::make_shared<runtime::PassthroughNoWaitDeviceTask>(wait_fn));
  } else {
    wait_fn();
  }
}

void CommHandle::Wait() {
  if (event_ != nullptr) {
    MS_LOG(DEBUG) << "CommHandle::Wait() called: event=" << event_ << ", op_name=" << op_name_
                  << ", CommHandle=" << this;
  }
  // Wait event async.
  WaitTaskFunc(shared_from_this(), CurrentStream::id());
  return;
}

void CommHandle::WaitForWaitCalled() const {
  constexpr int check_interval = 1000;
  constexpr int max_check_times = 10;
  std::chrono::milliseconds timeout(check_interval);
  std::unique_lock<std::mutex> lock(wait_called_mutex_);
  for (int i = 0; i < max_check_times; i++) {
    try {
      MsException::Instance().CheckException();
    } catch (const std::exception &e) {
      MS_LOG(WARNING) << "CheckException failed with exception: " << e.what();
      return;
    }
    auto result = wait_called_cv_.wait_for(lock, timeout, [this] { return wait_called_.load(); });
    if (!result) {
      MS_LOG(WARNING) << "WaitForWaitCalled timed out after " << timeout.count()
                      << " seconds. wait_called_=" << wait_called_.load() << ", op_name=" << op_name_
                      << ", CommHandle=" << this;
    }
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
