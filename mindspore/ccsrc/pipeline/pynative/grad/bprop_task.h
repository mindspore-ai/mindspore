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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_BPROP_TASK_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_BPROP_TASK_H_

#include <functional>
#include <utility>
#include "runtime/pipeline/task/task.h"

namespace mindspore {
namespace pynative {
class BpropTask : public runtime::AsyncTask {
 public:
  explicit BpropTask(const std::function<void(void)> &task) : AsyncTask(runtime::kBpropTask), run_task_(task) {}
  explicit BpropTask(std::function<void(void)> &&task) : AsyncTask(runtime::kBpropTask), run_task_(std::move(task)) {}
  ~BpropTask() override = default;
  void Run() override;

 private:
  std::function<void(void)> run_task_;
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_BPROP_TASK_H_
