/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "pybind_api/pynative/tensor/storage/storage_base.h"
#include <utility>
#include <memory>
#include <string>
#include "runtime/hardware_abstract/utils.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "runtime/pipeline/task/task.h"
#include "include/pynative/utils/pynative_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "mindspore/core/include/utils/stream_guard.h"

namespace mindspore {
namespace {
device::DeviceAddressPtr CreateTempDeviceAddress(const device::DeviceAddressPtr &device_address) {
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_address->GetDeviceType(), device_address->device_id()});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();
  auto new_device_address = std::make_shared<device::DeviceAddress>(
    device_address->GetMutablePtr(), device_address->size(), device_address->GetDeviceType(), CurrentStream::id());
  new_device_address->set_from_mem_pool(false);
  return new_device_address;
}

class StorageCopyTask : public runtime::AsyncTask {
 public:
  explicit StorageCopyTask(std::function<void(void)> run_func)
      : AsyncTask(runtime::kFrontendTask), run_func_(std::move(run_func)) {}
  explicit StorageCopyTask(std::function<void(void)> run_func, std::function<void()> set_exception_func)
      : AsyncTask(runtime::kFrontendTask),
        run_func_(std::move(run_func)),
        set_exception_func_(std::move(set_exception_func)) {}
  ~StorageCopyTask() override = default;
  void Run() override { run_func_(); };
  void SetException(const std::exception_ptr &e) override {
    if (set_exception_func_ == nullptr) {
      MS_LOG(ERROR) << "set_exception_func_ is null";
      return;
    }
    set_exception_func_();
  };

 private:
  std::function<void(void)> run_func_;
  std::function<void()> set_exception_func_;
};
};  // namespace

StorageBase::~StorageBase() { device_data_ = nullptr; }

uintptr_t StorageBase::DataPtr() const {
  MS_EXCEPTION_IF_NULL(device_data_);
  auto *data_ptr = device_data_->GetMutablePtr();
  return reinterpret_cast<uintptr_t>(data_ptr);
}

void StorageBase::InplaceReSize(int64_t size) {
  runtime::Pipeline::Get().WaitForward();
  MS_EXCEPTION_IF_NULL(device_data_);
  if (size == 0) {
    device_data_->ClearDeviceMemory();
    device_data_->SetSize(0);
    return;
  }

  device::DeviceContextKey host_key = {device_data_->GetDeviceType(), device_data_->device_id()};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  void *device_ptr = nullptr;
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "ResizeStorage", "ResizeStorage", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "ResizeStorage", memory::mem_pool::MemType::kOther, size,
                                                 device_data_.get());
  device_ptr = host_context->device_res_manager_->AllocateMemory(size, CurrentStream::id());
  if (!device_ptr) {
    return;
  }
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, device_data_.get(), device_ptr);
  device_data_->set_ptr(device_ptr);
  device_data_->set_from_mem_pool(true);
  device_data_->SetSize(size);
  device_data_->set_stream_id(CurrentStream::id());
}

int64_t StorageBase::NBytes() const {
  MS_EXCEPTION_IF_NULL(device_data_);
  return device_data_->size();
}

TypeId StorageBase::GetTypeId() const { return type_id_; }

uint32_t StorageBase::GetStreamId() const {
  MS_EXCEPTION_IF_NULL(device_data_);
  return device_data_->stream_id();
}

const DevicePointerPtr &StorageBase::GetDevicePointer() const {
  MS_EXCEPTION_IF_NULL(device_data_);
  return device_data_->device_pointer();
}

const DeviceAddressPtr &StorageBase::GetDeviceAddress() const {
  MS_EXCEPTION_IF_NULL(device_data_);
  return device_data_;
}

void StorageBase::InplaceCopy(const StorageBasePtr &src, bool non_blocking) {
  MS_EXCEPTION_IF_NULL(device_data_);
  MS_EXCEPTION_IF_NULL(src->device_data_);
  pynative::DispatchOp(std::make_shared<StorageCopyTask>(
    [dst_address = device_data_, src_address = src->device_data_, non_blocking = non_blocking]() {
      device::DeviceAddressPtr dst = CreateTempDeviceAddress(dst_address);
      device::DeviceAddressPtr src = CreateTempDeviceAddress(src_address);
      ShapeVector src_shape = {static_cast<int64_t>(src->size())};
      auto src_tensor = tensor::from_spec(kNumberTypeUInt8, src_shape, device::DeviceType::kNone);
      src_tensor->set_device_address(src);
      ShapeVector dst_shape = {static_cast<int64_t>(dst->size())};
      auto dst_tensor = tensor::from_spec(kNumberTypeUInt8, dst_shape, device::DeviceType::kNone);
      dst_tensor->set_device_address(dst);
      auto non_blocking_value = std::make_shared<mindspore::BoolImm>(non_blocking);
      // Fix inplace_copy. In recompute task, the rng state will be set, which cause the device_target to be cpu,
      // then the inplace_copy operator will not be dispatched Ascend. So reset the device_target of OpStatus here.
      if (device::IsAscendDeviceType(src_tensor->device_address()->GetDeviceType()) ||
          device::IsAscendDeviceType(dst_tensor->device_address()->GetDeviceType())) {
        kernel::pyboost::OpRunStatus::Get().set_run_info(kernel::pyboost::OpStatus(true, device::DeviceType::kAscend));
      }
      kernel::pyboost::inplace_copy(dst_tensor, src_tensor, non_blocking_value);
      (void)kernel::pyboost::OpRunStatus::Get().GetLastOp();
    }));
}

std::string StorageBase::device() const {
  MS_EXCEPTION_IF_NULL(device_data_);
  return device::GetDeviceNameByType(device_data_->GetDeviceType());
}

const MapAllocatorPtr &StorageBase::GetMapAllocator() const {
  MS_EXCEPTION_IF_NULL(device_data_);
  return device_data_->map_allocator();
}

void StorageBase::SetDevicePointer(const DevicePointerPtr device_pointer) {
  device_data_->set_device_pointer(device_pointer);
}

}  // namespace mindspore
