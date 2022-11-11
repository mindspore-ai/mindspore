/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "include/api/cell.h"
#include "include/api/context.h"
#include "cxx_api/factory.h"
#include "cxx_api/graph/graph_impl.h"

namespace mindspore {
std::vector<Output> CellBase::operator()(const std::vector<Input> &inputs) const { return Clone()->Construct(inputs); }

GraphCell::GraphCell(const Graph &graph) : graph_(std::make_shared<Graph>(graph)) { MS_EXCEPTION_IF_NULL(graph_); }

GraphCell::GraphCell(const std::shared_ptr<Graph> &graph) : graph_(graph) { MS_EXCEPTION_IF_NULL(graph_); }

GraphCell::GraphCell(Graph &&graph) : graph_(std::make_shared<Graph>(graph)) { MS_EXCEPTION_IF_NULL(graph_); }

void GraphCell::SetContext(const std::shared_ptr<Context> &context) {
  if (executor_ == nullptr) {
    executor_ = GraphImplFactory::Instance().Create(g_device_target);
    if (executor_ == nullptr) {
      MS_LOG(ERROR) << "Create graph impl for device target " << g_device_target << " failed.";
      return;
    }
    executor_->SetGraph(graph_);
  }
  executor_->SetContext(context);
}

Status GraphCell::Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  if (executor_ == nullptr) {
    executor_ = GraphImplFactory::Instance().Create(g_device_target);
    if (executor_ == nullptr) {
      MS_LOG(ERROR) << "Create graph impl for device target " << g_device_target << " failed.";
      return kMEFailed;
    }
    executor_->SetGraph(graph_);
  }
  return executor_->Run(inputs, outputs);
}

Status GraphCell::Load(uint32_t device_id) {
  if (executor_ == nullptr) {
    executor_ = GraphImplFactory::Instance().Create(g_device_target);
    if (executor_ == nullptr) {
      MS_LOG(ERROR) << "Create graph impl for device target " << g_device_target << " failed.";
      return kMEFailed;
    }
    executor_->SetGraph(graph_);
  }
  return executor_->Load(device_id);
}

std::vector<MSTensor> GraphCell::GetInputs() {
  if (executor_ == nullptr) {
    executor_ = GraphImplFactory::Instance().Create(g_device_target);
    if (executor_ == nullptr) {
      MS_LOG(ERROR) << "Create graph impl for device target " << g_device_target << " failed.";
      return {};
    }
    executor_->SetGraph(graph_);
  }
  return executor_->GetInputs();
}

std::vector<MSTensor> GraphCell::GetOutputs() {
  if (executor_ == nullptr) {
    executor_ = GraphImplFactory::Instance().Create(g_device_target);
    if (executor_ == nullptr) {
      MS_LOG(ERROR) << "Create graph impl for device target " << g_device_target << " failed.";
      return {};
    }
    executor_->SetGraph(graph_);
  }
  return executor_->GetOutputs();
}

InputAndOutput::InputAndOutput() : cell_(nullptr), prev_(), index_(-1) {}

InputAndOutput::InputAndOutput(const std::shared_ptr<CellBase> &cell, const std::vector<InputAndOutput> &prev,
                               int32_t index)
    : cell_(cell), prev_(prev), index_(index) {}
}  // namespace mindspore
