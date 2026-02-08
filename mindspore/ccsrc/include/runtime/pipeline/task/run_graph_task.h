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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIME_PIPELINE_TASK_RUN_GRAPH_TASK_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIME_PIPELINE_TASK_RUN_GRAPH_TASK_H_

#include <functional>
#include <memory>
#include <utility>

#include "include/utils/stub_tensor.h"
#include "runtime/pipeline/task/task.h"
#include "runtime/pipeline/visible.h"

namespace mindspore {
namespace runtime {
class RUNTIME_PIPELINE_EXPORT RunGraphTask : public AsyncTask {
 public:
  explicit RunGraphTask(std::function<void()> &&func, const stub::StubNodePtr &stub_output)
      : AsyncTask(kRunGraphTask), func_(std::move(func)), stub_output_(stub_output) {}
  ~RunGraphTask() override = default;
  void Run() override;
  void SetException(const std::exception_ptr &e) override;

 protected:
  std::function<void()> func_;
  stub::StubNodePtr stub_output_;
};
using RunGraphTaskPtr = std::shared_ptr<RunGraphTask>;
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIME_PIPELINE_TASK_RUN_GRAPH_TASK_H_
