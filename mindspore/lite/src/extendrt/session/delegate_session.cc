/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <vector>
#include <string>
#include <memory>

#include "extendrt/session/delegate_session.h"
#include "extendrt/session/graph_executor_session.h"
#include "extendrt/session/factory.h"
#include "extendrt/delegate/graph_executor/delegate.h"

namespace mindspore {
Status DelegateSession::Init(const std::shared_ptr<Context> context) { return kSuccess; }
Status DelegateSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size) { return kSuccess; }

Status DelegateSession::RunGraph() { return kSuccess; }
Status DelegateSession::RunGraph(const std::vector<tensor::TensorPtr> &inputs,
                                 std::vector<tensor::TensorPtr> *outputs) {
  return kSuccess;
}
Status DelegateSession::Resize(const std::vector<tensor::TensorPtr> &inputs,
                               const std::vector<std::vector<int64_t>> &dims) {
  return kSuccess;
}
std::vector<tensor::TensorPtr> DelegateSession::GetOutputs() { return std::vector<tensor::TensorPtr>(); }
std::vector<tensor::TensorPtr> DelegateSession::GetInputs() { return std::vector<tensor::TensorPtr>(); }
std::vector<std::string> DelegateSession::GetOutputNames() { return std::vector<std::string>(); }
std::vector<std::string> DelegateSession::GetInputNames() { return std::vector<std::string>(); }
tensor::TensorPtr DelegateSession::GetOutputByTensorName(const std::string &tensorName) { return nullptr; }
tensor::TensorPtr DelegateSession::GetInputByTensorName(const std::string &name) { return nullptr; }

static std::shared_ptr<InferSession> DelegateSessionCreator(const SessionConfig &config) {
  auto delegates = config.delegates_;
  if (delegates.size() > 1) {
    MS_LOG(ERROR) << "Not support multi delegates context";
    return nullptr;
  }
  auto delegate = delegates.front();

  auto graph_executor_delegate = std::reinterpret_pointer_cast<GraphExecutorDelegate>(delegate);
  if (graph_executor_delegate != nullptr) {
    return std::make_shared<GraphExecutorSession>(graph_executor_delegate->GetGraphExecutor());
  }

  return std::make_shared<DelegateSession>(delegate);
}

REG_SESSION(kDelegateSession, DelegateSessionCreator);
}  // namespace mindspore
