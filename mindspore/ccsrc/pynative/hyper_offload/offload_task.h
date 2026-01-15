/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_TASK_H_
#define MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_TASK_H_

#include "include/runtime/pipeline/pipeline.h"
#include "runtime/pipeline/task/task.h"
#include "include/pynative/utils/runtime/op_executor.h"

namespace mindspore {
namespace pynative {
namespace hyper_offload {

// AsyncTask for offload operations
class OffloadAsyncTask : public runtime::AsyncTask {
 public:
  explicit OffloadAsyncTask(std::function<void()> func, const runtime::TaskType &task)
      : runtime::AsyncTask(task), func_(std::move(func)) {}
  ~OffloadAsyncTask() override = default;

  void Run() override {
    if (func_) {
      func_();
    }
  }

 private:
  std::function<void()> func_;
};
}  // namespace hyper_offload
}  // namespace pynative
}  // namespace mindspore

#endif  //  MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_TASK_H_
