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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_RUNTIME_DEFAULT_GRAPH_RUNTIME_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_RUNTIME_DEFAULT_GRAPH_RUNTIME_H_

#include <vector>
#include <memory>
#include <string>

#include "infer/executor.h"
#include "infer/execution_plan.h"
#include "litert/executor.h"

namespace mindspore {
class MindRTGraphExecutor : public mindspore::infer::abstract::Executor {
 public:
  MindRTGraphExecutor();
  explicit MindRTGraphExecutor(const std::string &name, std::shared_ptr<infer::abstract::ExecutionPlan> execution_plan);
  virtual ~MindRTGraphExecutor() = default;

  const std::string &Name() override { return name_; }

  Status Prepare() override;

  Status Execute() override;

  int Resize(const std::vector<infer::abstract::Tensor *> &inputs,
             const std::vector<std::vector<int64_t>> &dims) override;

 private:
  std::string name_;
  std::shared_ptr<mindspore::lite::Executor> mindrt_executor_;
  std::shared_ptr<infer::abstract::ExecutionPlan> execution_plan_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_RUNTIME_DEFAULT_GRAPH_RUNTIME_H_
