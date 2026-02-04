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

#include "runtime/pipeline/task/task.h"

namespace mindspore {
namespace runtime {

class KernelTaskContext {
 public:
  KernelTaskContext(const device::DeviceContext *device_context, tensor::TensorPtrList input_tensors,
                    tensor::TensorPtrList output_tensors, void *stream)
      : device_context_(device_context),
        input_tensors_(std::move(input_tensors)),
        output_tensors_(std::move(output_tensors)),
        stream_(stream) {}
  ~KernelTaskContext() = default;

  const device::DeviceContext *device_context() { return device_context_; }
  void *stream() { return stream_; }

  const tensor::TensorPtr GetInput(size_t idx) {
    if (idx >= input_tensors_.size()) {
      MS_LOG(EXCEPTION) << "input_tensors size is invalid, size:" << input_tensors_.size() << ", idx:" << idx;
    }
    auto tensor = input_tensors_[idx];
    MS_EXCEPTION_IF_NULL(tensor);
    return tensor;
  }

  const tensor::TensorPtr GetOutput(size_t idx) {
    if (idx >= output_tensors_.size()) {
      MS_LOG(EXCEPTION) << "output_tensors size is invalid, size:" << output_tensors_.size() << ", idx:" << idx;
    }
    auto tensor = output_tensors_[idx];
    MS_EXCEPTION_IF_NULL(tensor);
    return tensor;
  }

 private:
  const device::DeviceContext *device_context_;
  tensor::TensorPtrList input_tensors_;
  tensor::TensorPtrList output_tensors_;
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

}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_KERNEL_TASK_H_
