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

#include "extendrt/graph_executor/plan_executor.h"
#include "extendrt/execution_plan.h"
#include "litert/mindrt_executor.h"

namespace mindspore::infer {
PlanExecutor::PlanExecutor() { PlanExecutor("PlanExecutor"); }

PlanExecutor::PlanExecutor(const std::string &name, std::shared_ptr<abstract::ExecutionPlan> execution_plan) {
  name_ = name;
  execution_plan_ = execution_plan;
  auto infer_execution_plan = std::dynamic_pointer_cast<infer::ExecutionPlan>(execution_plan_);
  if (infer_execution_plan == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::FlowExecutor Not Supported execution plan is passed";
  } else {
    executor_ = std::make_shared<mindspore::lite::MindrtExecutor>(infer_execution_plan->GetInputMap(),
                                                                  infer_execution_plan->GetOutputMap());
  }
}

Status PlanExecutor::Prepare() {
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::Prepare executor is nullptr";
    return kLiteError;
  }

  if (execution_plan_ == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::Prepare execution plan is nullptr";
    return kLiteError;
  }
  return executor_->Prepare(execution_plan_->ToKernelList(), execution_plan_->GetInputs(),
                            execution_plan_->GetOutputs(), execution_plan_->GetContext());
}

Status PlanExecutor::Execute() {
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::Execute executor is nullptr";
    return kLiteError;
  }
  if (execution_plan_ == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::Execute execution plan is nullptr";
    return kLiteError;
  }
  return executor_->Run(execution_plan_->GetInputs(), execution_plan_->GetOutputs(), execution_plan_->ToKernelList(),
                        execution_plan_->GetKernelBeforeCallBack(), execution_plan_->GetKernelAfterCallBack());
}

int PlanExecutor::Resize(const std::vector<abstract::Tensor *> &inputs, const std::vector<std::vector<int>> &dims) {
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::Resize executor is nullptr";
    return kLiteError;
  }
  return executor_->Resize(inputs, dims);
}
}  // namespace mindspore::infer
