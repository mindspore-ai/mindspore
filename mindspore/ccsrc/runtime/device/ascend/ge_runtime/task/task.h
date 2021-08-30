/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_TASK_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_TASK_H_

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "runtime/device/ascend/ge_runtime/model_context.h"
#include "runtime/device/ascend/ge_runtime/task_info.h"

namespace mindspore::ge::model_runner {
class Task {
 public:
  Task() {}

  virtual ~Task() {}

  virtual void Distribute() = 0;

  virtual void *Args() { return nullptr; }

  virtual std::string task_name() const { return ""; }

  void set_model_handle(rtModel_t model_handle) { model_handle_ = model_handle; }

 protected:
  rtModel_t model_handle_{nullptr};
};

template <class T>
class TaskRepeater : public Task {
  static_assert(std::is_base_of<TaskInfo, T>(), "Wrong TaskInfo Type!");

 public:
  TaskRepeater(const ModelContext &model_context, const std::shared_ptr<T> &task_info) {}

  virtual ~TaskRepeater() {}

  virtual void Distribute() = 0;
};
}  // namespace mindspore::ge::model_runner
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_TASK_H_
