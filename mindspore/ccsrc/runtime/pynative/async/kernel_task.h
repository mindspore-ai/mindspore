/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_KERNEL_TASK_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_KERNEL_TASK_H_

#include <utility>
#include <vector>
#include <memory>
#include <future>

#include "runtime/pynative/async/task.h"

namespace mindspore {
namespace pynative {

class KernelTaskContext {
 public:
  KernelTaskContext(const device::DeviceContext *device_context, device::DeviceAddressPtrList input_addr_list,
                    TensorStorageInfoPtrList input_storage_list, device::DeviceAddressPtrList output_addr_list,
                    void *stream)
      : device_context_(device_context),
        input_addr_list_(std::move(input_addr_list)),
        output_addr_list_(std::move(output_addr_list)),
        input_storage_list_(std::move(input_storage_list)),
        stream_(stream) {}
  ~KernelTaskContext() = default;

  const device::DeviceContext *device_context() { return device_context_; }
  void *stream() { return stream_; }

  const device::DeviceAddressPtr GetInputAddr(size_t idx) {
    if (idx >= input_addr_list_.size()) {
      MS_LOG(EXCEPTION) << "input_addr_list size is invalid, size:" << input_addr_list_.size() << ", idx:" << idx;
    }
    auto addr = input_addr_list_[idx];
    MS_EXCEPTION_IF_NULL(addr);
    return addr;
  }

  const device::DeviceAddressPtr GetOutputAddr(size_t idx) {
    if (idx >= output_addr_list_.size()) {
      MS_LOG(EXCEPTION) << "output_addr_list_ size is invalid, size:" << output_addr_list_.size() << ", idx:" << idx;
    }
    auto addr = output_addr_list_[idx];
    MS_EXCEPTION_IF_NULL(addr);
    return addr;
  }

  const TensorStorageInfoPtr GetInputStorage(size_t idx) {
    if (idx >= input_storage_list_.size()) {
      MS_LOG(EXCEPTION) << "input_storage_list_ size is invalid, size:" << input_storage_list_.size()
                        << ", idx:" << idx;
    }
    auto addr = input_storage_list_[idx];
    return addr;
  }

 private:
  const device::DeviceContext *device_context_;
  device::DeviceAddressPtrList input_addr_list_;
  device::DeviceAddressPtrList output_addr_list_;
  TensorStorageInfoPtrList input_storage_list_;
  void *stream_;
};

class KernelTask : public AsyncTask {
 public:
  explicit KernelTask(std::shared_ptr<KernelTaskContext> context)
      : AsyncTask(kKernelTask), context_(std::move(context)) {}
  ~KernelTask() override = default;
  void Run() override {}

 protected:
  std::shared_ptr<KernelTaskContext> context_;
};
using KernelTaskPtr = std::shared_ptr<KernelTask>;

}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_KERNEL_TASK_H_
