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
#include "extendrt/graph_runtime/default_graph_runtime.h"

#include "extendrt/graph_runtime/factory.h"
#include "extendrt/graph_executor/factory.h"

namespace mindspore {
using ExecutionPlan = mindspore::infer::abstract::ExecutionPlan;

Status DefaultGraphRuntime::Prepare(std::shared_ptr<ExecutionPlan> execution_plan) {
  MS_LOG(INFO) << "DefaultGraphRuntime::Prepare Begin";

  if (execution_plan == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Prepare Execution Plan is nullptr.";
    return kLiteNullptr;
  }
  execution_plan_ = execution_plan;

  auto executor = SelectExecutor();
  if (executor == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Prepare Select Executor is nullptr.";
    return kLiteNullptr;
  }

  MS_LOG(DEBUG) << "DefaultGraphRuntime::Prepare Prepare Execution Plan Begin of Executor " << executor->Name();
  auto status = executor->Prepare();
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Prepare Prepare Execution Plan Failed in Executor " << executor->Name();
    return kLiteError;
  }
  MS_LOG(DEBUG) << "DefaultGraphRuntime::Prepare Prepare Execution Plan End";

  MS_LOG(INFO) << "AbstractRuntime::Prepare End";
  return kSuccess;
}

Status DefaultGraphRuntime::Execute() {
  MS_LOG(INFO) << "DefaultGraphRuntime::Execute Begin";

  if (execution_plan_ == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Execution Plan is nullptr.";
    return kLiteNullptr;
  }

  auto executor = SelectExecutor();
  if (executor == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Select Executor is nullptr.";
    return kLiteNullptr;
  }

  MS_LOG(DEBUG) << "DefaultGraphRuntime::Execute Execute Execution Plan Begin of Executor " << executor->Name();
  auto status = executor->Execute();
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Execute Execution Plan Failed in Executor " << executor->Name();
    return kLiteError;
  }
  MS_LOG(DEBUG) << "DefaultGraphRuntime::Execute Execute Execution Plan End";

  MS_LOG(INFO) << "DefaultGraphRuntime::Execute End";
  return kSuccess;
}

Status DefaultGraphRuntime::Execute(const std::vector<infer::abstract::Tensor *> &inputs,
                                    const std::vector<infer::abstract::Tensor *> &outputs,
                                    infer::abstract::KernelCallBack before, infer::abstract::KernelCallBack after) {
  MS_LOG(INFO) << "DefaultGraphRuntime::Execute Begin";

  if (execution_plan_ == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Execution Plan is nullptr.";
    return kLiteNullptr;
  }

  auto executor = SelectExecutor();
  if (executor == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Select Executor is nullptr.";
    return kLiteNullptr;
  }

  MS_LOG(DEBUG) << "DefaultGraphRuntime::Execute Execute Execution Plan Begin of Executor " << executor->Name();
  execution_plan_->SetInputs(inputs);
  execution_plan_->SetOutputs(outputs);
  execution_plan_->SetKernelBeforeCallBack(before);
  execution_plan_->SetKernelAfterCallBack(after);
  auto status = executor->Execute();
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Execute Execution Plan Failed in Executor " << executor->Name();
    return kLiteError;
  }
  MS_LOG(DEBUG) << "DefaultGraphRuntime::Execute Execute Execution Plan End";

  MS_LOG(INFO) << "DefaultGraphRuntime::Execute End";
  return kSuccess;
}

Status DefaultGraphRuntime::Resize(const std::vector<infer::abstract::Tensor *> &inputs,
                                   const std::vector<std::vector<int64_t>> &dims) {
  MS_LOG(INFO) << "DefaultGraphRuntime::Resize Begin";

  if (execution_plan_ == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Resize Execution Plan is nullptr.";
    return kLiteNullptr;
  }

  auto executor = SelectExecutor();
  if (executor == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Resize Select Executor is nullptr.";
    return kLiteNullptr;
  }

  MS_LOG(DEBUG) << "DefaultGraphRuntime::Resize Resize Execution Plan Begin of Executor " << executor->Name();
  auto status = executor->Resize(inputs, dims);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Resize Resize Execution Plan Failed in Executor " << executor->Name();
    return kLiteError;
  }
  MS_LOG(DEBUG) << "DefaultGraphRuntime::Resize Resize Execution Plan End";

  MS_LOG(INFO) << "DefaultGraphRuntime::Resize End";
  return kSuccess;
}

std::vector<infer::abstract::Tensor *> DefaultGraphRuntime::GetInputs() {
  if (execution_plan_ == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Execution Plan is nullptr.";
    return std::vector<infer::abstract::Tensor *>{};
  }
  return execution_plan_->GetInputs();
}

std::vector<infer::abstract::Tensor *> DefaultGraphRuntime::GetOutputs() {
  if (execution_plan_ == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Execution Plan is nullptr.";
    return std::vector<infer::abstract::Tensor *>{};
  }
  return execution_plan_->GetOutputs();
}

std::shared_ptr<infer::abstract::Executor> DefaultGraphRuntime::SelectExecutor() {
  if (default_executor_ == nullptr) {
    default_executor_ =
      GraphExecutorRegistry::GetInstance().GetExecutor(kMindRTExecutor, "mindrt-executor", execution_plan_);
  }
  return default_executor_;
}

static std::shared_ptr<infer::abstract::GraphRuntime> DefaultGraphRuntimeCreator() {
  auto graph_runtime = std::make_shared<DefaultGraphRuntime>();
  return graph_runtime;
}
REG_GRAPH_RUNTIME(kDefaultRuntime, DefaultGraphRuntimeCreator);
}  // namespace mindspore
