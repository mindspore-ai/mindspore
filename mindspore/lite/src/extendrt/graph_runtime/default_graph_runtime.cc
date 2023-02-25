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

#include "extendrt/flow_executor.h"
#include "src/common/log.h"

namespace mindspore {
using ExecutionPlan = mindspore::infer::abstract::ExecutionPlan;

Status DefaultGraphRuntime::Prepare(std::shared_ptr<ExecutionPlan> execution_plan) {
  MS_LOG(INFO) << "DefaultGraphRuntime::Prepare Begin";

  if (execution_plan == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Prepare Execution Plan is nullptr.";
    return kLiteNullptr;
  }
  execution_plan_ = execution_plan;

  for (auto execution_flow : execution_plan->GetExecutionFLows()) {
    auto executor = SelectExecutor(execution_flow);
    if (executor == nullptr) {
      MS_LOG(ERROR) << "DefaultGraphRuntime::Prepare Select Executor is nullptr.";
      return kLiteNullptr;
    }
    MS_LOG(DEBUG) << "DefaultGraphRuntime::Prepare Prepare Execution Plan Begin of Executor " << executor->Name();
    auto status = executor->Prepare(execution_flow);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "DefaultGraphRuntime::Prepare Prepare Execution Plan Failed in Executor " << executor->Name();
      return kLiteError;
    }
    MS_LOG(DEBUG) << "DefaultGraphRuntime::Prepare Prepare Execution Plan End";
  }
  MS_LOG(INFO) << "AbstractRuntime::Prepare End";
  return kSuccess;
}

Status DefaultGraphRuntime::Execute() {
  MS_LOG(INFO) << "DefaultGraphRuntime::Execute Begin";

  if (execution_plan_ == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Execution Plan is nullptr.";
    return kLiteNullptr;
  }

  for (auto execution_flow : execution_plan_->GetExecutionFLows()) {
    auto executor = SelectExecutor(execution_flow);
    if (executor == nullptr) {
      MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Select Executor is nullptr.";
      return kLiteNullptr;
    }
    MS_LOG(DEBUG) << "DefaultGraphRuntime::Execute Execution Plan Begin of Executor " << executor->Name();
    auto status = executor->Execute();
    if (status != kSuccess) {
      MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Execution Plan Failed in Executor " << executor->Name();
      return kLiteError;
    }
    MS_LOG(DEBUG) << "DefaultGraphRuntime::Execute Prepare Execution Plan End";
  }
  MS_LOG(INFO) << "DefaultGraphRuntime::Execute End";
  return kSuccess;
}

Status DefaultGraphRuntime::Execute(const std::vector<abstract::Tensor *> &inputs,
                                    const std::vector<abstract::Tensor *> &outputs, abstract::KernelCallBack before,
                                    abstract::KernelCallBack after) {
  MS_LOG(INFO) << "DefaultGraphRuntime::Execute Begin";

  if (execution_plan_ == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Execution Plan is nullptr.";
    return kLiteNullptr;
  }

  for (auto &execution_flow : execution_plan_->GetExecutionFLows()) {
    auto executor = SelectExecutor(execution_flow);
    if (executor == nullptr) {
      MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Select Executor is nullptr.";
      return kLiteNullptr;
    }
    MS_LOG(DEBUG) << "DefaultGraphRuntime::Execute Execution Plan Begin of Executor " << executor->Name();
    execution_flow->SetInputs(inputs);
    execution_flow->SetOutputs(outputs);
    execution_flow->SetKernelBeforeCallBack(before);
    execution_flow->SetKernelAfterCallBack(after);
    auto status = executor->Execute();
    if (status != kSuccess) {
      MS_LOG(ERROR) << "DefaultGraphRuntime::Execute Execution Plan Failed in Executor " << executor->Name();
      return kLiteError;
    }
    MS_LOG(DEBUG) << "DefaultGraphRuntime::Execute Prepare Execution Plan End";
  }
  MS_LOG(INFO) << "DefaultGraphRuntime::Execute End";
  return kSuccess;
}

std::shared_ptr<abstract::Executor> DefaultGraphRuntime::SelectExecutor(
  const std::shared_ptr<abstract::ExecutionFlow> &execution_flow) {
  auto it = executor_map_.find(execution_flow);
  if (it == executor_map_.end()) {
    // create a new executor for execution flow
    auto executor = std::make_shared<infer::FlowExecutor>("flow-executor");
    executor_map_[execution_flow] = executor;
    return executor;
  }
  return it->second;
}
}  // namespace mindspore
