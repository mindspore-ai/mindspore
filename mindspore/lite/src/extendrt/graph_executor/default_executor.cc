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

#include "extendrt/graph_executor/default_executor.h"

#include <memory>
#include "extendrt/graph_executor/factory.h"
#include "extendrt/execution_plan.h"
#include "litert/mindrt_executor.h"

namespace mindspore {
DefaultExecutor::DefaultExecutor() {
  name_ = "DefaultExecutor";
  execution_plan_ = nullptr;
}

DefaultExecutor::DefaultExecutor(const std::string &name,
                                 std::shared_ptr<infer::abstract::ExecutionPlan> execution_plan) {
  name_ = name;
  execution_plan_ = execution_plan;
}

bool DefaultExecutor::Init() {
  auto infer_execution_plan = std::dynamic_pointer_cast<infer::ExecutionPlan>(execution_plan_);
  if (infer_execution_plan == nullptr) {
    MS_LOG(ERROR) << "Not Supported execution plan is passed";
    return false;
  }
  if (!infer_execution_plan->PrepareKernels()) {
    MS_LOG(ERROR) << "Prepare kernels failed";
    return false;
  }

  inited_ = true;
  return true;
}

Status DefaultExecutor::Prepare() {
  if (!Init()) {
    MS_LOG(ERROR) << "Init executor failed";
    return kLiteError;
  }

  MS_ASSERT(inited_ == true);
  MS_ASSERT(execution_plan_ != nullptr);

  // only single sub graph will choose this executor
  MS_ASSERT(execution_plan_->ToKernelList().size() == 1);

  auto sub_graph_kernel = execution_plan_->ToKernelList().at(0);
  if (sub_graph_kernel == nullptr) {
    MS_LOG(ERROR) << "Sub graph kernel is nullptr";
    return kLiteNullptr;
  }

  if (sub_graph_kernel->Prepare() != RET_OK) {
    MS_LOG(ERROR) << "Sub graph kernel prepare failed";
    return kLiteError;
  }

  return kSuccess;
}

Status DefaultExecutor::Execute() {
  if (!inited_) {
    MS_LOG(ERROR) << "Executor is not inited correctly";
    return kLiteError;
  }
  MS_ASSERT(execution_plan_ != nullptr);

  // only single sub graph will choose this executor
  MS_ASSERT(execution_plan_->ToKernelList().size() == 1);

  auto sub_graph_kernel = execution_plan_->ToKernelList().at(0);
  if (sub_graph_kernel == nullptr) {
    MS_LOG(ERROR) << "Sub graph kernel is nullptr";
    return kLiteNullptr;
  }

  // copy data to sub_graph_inputs
  auto sub_graph_inputs = sub_graph_kernel->in_tensors();
  auto user_inputs = execution_plan_->GetInputs();
  for (size_t i = 0; i < user_inputs.size(); ++i) {
    auto sub_graph_input = sub_graph_inputs.at(i);
    auto user_input = user_inputs.at(i);
    sub_graph_input->set_data(user_input->data());
    sub_graph_input->set_category(lite::GRAPH_INPUT);
  }

  // copy data to sub_graph_outputs
  auto sub_graph_outputs = sub_graph_kernel->out_tensors();
  auto user_outputs = execution_plan_->GetOutputs();
  for (size_t i = 0; i < user_outputs.size(); ++i) {
    auto sub_graph_output = sub_graph_outputs.at(i);
    auto user_output = user_outputs.at(i);
    sub_graph_output->set_data(user_output->data());
    sub_graph_output->set_category(lite::GRAPH_OUTPUT);
  }

  if (sub_graph_kernel->Execute() != RET_OK) {
    MS_LOG(ERROR) << "Sub graph kernel execute failed";
    return kLiteError;
  }

  for (auto sub_graph_input : sub_graph_inputs) {
    sub_graph_input->set_data(nullptr);
  }

  for (auto sub_graph_output : sub_graph_outputs) {
    sub_graph_output->set_data(nullptr);
  }

  return kSuccess;
}

int DefaultExecutor::Resize(const std::vector<infer::abstract::Tensor *> &inputs,
                            const std::vector<std::vector<int64_t>> &dims) {
  if (!inited_) {
    MS_LOG(ERROR) << "Executor is not inited correctly";
    return kLiteError;
  }
  MS_ASSERT(execution_plan_ != nullptr);

  // only single sub graph will choose this executor
  MS_ASSERT(execution_plan_->ToKernelList().size() == 1);

  auto sub_graph_kernel = execution_plan_->ToKernelList().at(0);
  if (sub_graph_kernel == nullptr) {
    MS_LOG(ERROR) << "Sub graph kernel is nullptr";
    return kLiteNullptr;
  }

  auto ret = sub_graph_kernel->ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Sub graph kernel resize failed";
    return kLiteError;
  }

  return RET_OK;
}

static std::shared_ptr<infer::abstract::Executor> DefaultGraphExecutorCreator(
  const std::string &name, std::shared_ptr<infer::abstract::ExecutionPlan> execution_plan) {
  auto graph_executor = std::make_shared<DefaultExecutor>(name, execution_plan);
  return graph_executor;
}
REG_GRAPH_EXECUTOR(kDefaultExecutor, DefaultGraphExecutorCreator);
}  // namespace mindspore
