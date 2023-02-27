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

#include "infer/graph_runtime.h"

namespace mindspore {
class DefaultGraphRuntime : public mindspore::infer::abstract::GraphRuntime {
 public:
  DefaultGraphRuntime() = default;
  virtual ~DefaultGraphRuntime() = default;

  Status Prepare(std::shared_ptr<infer::abstract::ExecutionPlan> execution_plan) override;

  Status Execute() override;

  Status Execute(const std::vector<infer::abstract::Tensor *> &inputs,
                 const std::vector<infer::abstract::Tensor *> &outputs,
                 infer::abstract::KernelCallBack before = nullptr,
                 infer::abstract::KernelCallBack after = nullptr) override;

  Status Resize(const std::vector<infer::abstract::Tensor *> *inputs,
                const std::vector<std::vector<int64_t>> &dims) override;

 private:
  std::shared_ptr<infer::abstract::Executor> SelectExecutor();

 private:
  std::shared_ptr<infer::abstract::ExecutionPlan> execution_plan_ = nullptr;
  std::shared_ptr<infer::abstract::Executor> default_executor_ = nullptr;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_RUNTIME_DEFAULT_GRAPH_RUNTIME_H_
