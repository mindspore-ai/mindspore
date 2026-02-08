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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIME_PIPELINE_TASK_BATCH_LAUNCH_KERNEL_TASK_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIME_PIPELINE_TASK_BATCH_LAUNCH_KERNEL_TASK_H_

#include <functional>
#include <memory>
#include <utility>

#include "runtime/pipeline/task/task.h"
#include "runtime/pipeline/visible.h"

namespace mindspore {
namespace runtime {
class RUNTIME_PIPELINE_EXPORT BatchLaunchKernelTask : public AsyncTask {
 public:
  explicit BatchLaunchKernelTask(std::function<void()> &&func) : AsyncTask(kKernelTask), func_(std::move(func)) {}
  ~BatchLaunchKernelTask() override = default;
  void Run() override;

 protected:
  std::function<void()> func_;
};
using BatchLaunchKernelTaskPtr = std::shared_ptr<BatchLaunchKernelTask>;
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIME_PIPELINE_TASK_BATCH_LAUNCH_KERNEL_TASK_H_
