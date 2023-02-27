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
#include "extendrt/graph_executor/mindrt_graph_executor.h"

#include "src/common/log.h"
#include "extendrt/graph_executor/factory.h"
#include "litert/mindrt_executor.h"
#include "extendrt/execution_plan.h"

namespace mindspore {
MindRTGraphExecutor::MindRTGraphExecutor() {
  name_ = "";
  execution_plan_ = nullptr;
}

MindRTGraphExecutor::MindRTGraphExecutor(const std::string &name,
                                         std::shared_ptr<infer::abstract::ExecutionPlan> execution_plan) {
  name_ = name;
  execution_plan_ = execution_plan;
  auto infer_execution_plan = std::dynamic_pointer_cast<infer::ExecutionPlan>(execution_plan_);
  if (infer_execution_plan == nullptr) {
    MS_LOG(ERROR) << "MindRTGraphExecutor::MindRTGraphExecutor Not Supported execution plan is passed";
  } else {
    mindrt_executor_ = std::make_shared<mindspore::lite::MindrtExecutor>(infer_execution_plan->GetInputMap(),
                                                                         infer_execution_plan->GetOutputMap());
  }
}

Status MindRTGraphExecutor::Prepare() {
  if (mindrt_executor_ == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::Prepare executor is nullptr";
    return kLiteError;
  }
  if (execution_plan_ == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::Prepare execution plan is nullptr";
    return kLiteError;
  }
  return mindrt_executor_->Prepare(execution_plan_->ToKernelList(), execution_plan_->GetInputs(),
                                   execution_plan_->GetOutputs(), execution_plan_->GetContext().get());
}

Status MindRTGraphExecutor::Execute() {
  if (mindrt_executor_ == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::Execute executor is nullptr";
    return kLiteError;
  }
  if (execution_plan_ == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::Execute execution plan is nullptr";
    return kLiteError;
  }
  return mindrt_executor_->Run(execution_plan_->GetInputs(), execution_plan_->GetOutputs(),
                               execution_plan_->ToKernelList(), execution_plan_->GetKernelBeforeCallBack(),
                               execution_plan_->GetKernelAfterCallBack());
}

int MindRTGraphExecutor::Resize(const std::vector<infer::abstract::Tensor *> &inputs,
                                const std::vector<std::vector<int>> &dims) {
  if (mindrt_executor_ == nullptr) {
    MS_LOG(ERROR) << "FlowExecutor::Resize executor is nullptr";
    return kLiteError;
  }
  return mindrt_executor_->Resize(inputs, dims);
}

static std::shared_ptr<infer::abstract::Executor> MindRTGraphExecutorCreator(
  const std::string &name, std::shared_ptr<infer::abstract::ExecutionPlan> execution_plan) {
  auto graph_executor = std::make_shared<MindRTGraphExecutor>(name, execution_plan);
  return graph_executor;
}
REG_GRAPH_EXECUTOR(kMindRTExecutor, MindRTGraphExecutorCreator);
}  // namespace mindspore
