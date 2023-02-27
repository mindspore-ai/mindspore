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

#include "extendrt/graph_executor/flow_executor.h"
#include "extendrt/execution_plan.h"
#include "litert/mindrt_executor.h"

namespace mindspore::infer {
FlowExecutor::FlowExecutor() { FlowExecutor("FlowExecutor"); }

FlowExecutor::FlowExecutor(const std::string &name, std::shared_ptr<abstract::ExecutionPlan> execution_plan) {
  name_ = name;
  execution_plan_ = execution_plan;
  auto infer_execution_plan = std::dynamic_pointer_cast<infer::ExecutionPlan>(execution_plan_);
  if (infer_execution_plan == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::FlowExecutor Not Supported execution plan is passed";
  } else {
    executor_ = std::make_shared<mindspore::lite::MindrtExecutor>(infer_execution_plan->GetInputMap(),
                                                                  infer_execution_plan->GetOutputMap);
  }
}

Status FlowExecutor::Prepare(std::shared_ptr<abstract::ExecutionFlow> execution_flow) {
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::Prepare executor is nullptr";
    return kLiteError;
  }

  if (execution_flow == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::Prepare execution flow is nullptr";
    return kLiteError;
  }

  return executor_->Prepare(execution_flow->GetKernels(), execution_flow->GetInputs(), execution_flow->GetOutputs(),
                            execution_flow->GetContext);
}

Status FlowExecutor::Execute() { return kSuccess; }

int FlowExecutor::Resize(const std::vector<abstract::Tensor *> &inputs, const std::vector<std::vector<int>> &dims) {
  return kSuccess;
}
}  // namespace mindspore::infer
