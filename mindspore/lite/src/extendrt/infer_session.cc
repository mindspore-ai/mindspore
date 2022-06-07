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
#include "extendrt/infer_session.h"

namespace mindspore {
class DefaultInferSession : public InferSession {
 public:
  DefaultInferSession() = default;
  virtual ~DefaultInferSession() = default;
  Status Init(const std::shared_ptr<Context> context) override;
  Status CompileGraph(FuncGraphPtr graph) override;
  Status RunGraph() override;
  Status Resize(const std::vector<tensor::TensorPtr> &inputs, const std::vector<std::vector<int64_t>> &dims) override;

  std::vector<tensor::TensorPtr> GetOutputs() override;
  std::vector<tensor::TensorPtr> GetInputs() override;
  tensor::TensorPtr GetOutputByTensorName(const std::string &tensorName) override;
  tensor::TensorPtr GetInputByTensorName(const std::string &name) override;
};

Status DefaultInferSession::Init(const std::shared_ptr<Context> context) { return kSuccess; }
Status DefaultInferSession::CompileGraph(FuncGraphPtr graph) { return kSuccess; }
Status DefaultInferSession::RunGraph() { return kSuccess; }
Status DefaultInferSession::Resize(const std::vector<tensor::TensorPtr> &inputs,
                                   const std::vector<std::vector<int64_t>> &dims) {
  return kSuccess;
}
std::vector<tensor::TensorPtr> DefaultInferSession::GetOutputs() { return std::vector<tensor::TensorPtr>(); }
std::vector<tensor::TensorPtr> DefaultInferSession::GetInputs() { return std::vector<tensor::TensorPtr>(); }
tensor::TensorPtr DefaultInferSession::GetOutputByTensorName(const std::string &tensorName) { return nullptr; }
tensor::TensorPtr DefaultInferSession::GetInputByTensorName(const std::string &name) { return nullptr; }
std::shared_ptr<InferSession> InferSession::CreateSession(const std::shared_ptr<Context> context) {
  return std::make_shared<DefaultInferSession>();
}
}  // namespace mindspore
