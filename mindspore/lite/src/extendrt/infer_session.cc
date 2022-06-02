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
Status InferSession::Init(const std::shared_ptr<Context> context) {
  basic_ = std::make_shared<SessionBasic>();
  return kSuccess;
}
std::shared_ptr<InferSession> InferSession::CreateSession(const std::shared_ptr<Context> context) {
  return std::make_shared<InferSession>();
}
int InferSession::CompileGraph(FuncGraphPtr graph) { return basic_->CompileGraph(graph); }
int InferSession::RunGraph() {}
tensor::TensorPtr InferSession::GetOutputByTensorName(const std::string &tensorName) {}
std::vector<tensor::TensorPtr> InferSession::GetOutputs() {}
Status InferSession::Resize(const std::vector<Tensor::TensorPtr> &inputs,
                            const std::vector<std::vector<int64_t>> &dims) {}
std::vector<Tensor::TensorPtr> InferSession::GetInputs() {}
std::vector<Tensor::TensorPtr> InferSession::GetOutputs();
Tensor::TensorPtr InferSession::GetInputByTensorName(const std::string &name) {}
std::vector<std::string> InferSession::GetOutputTensorNames() {}
virtual Tensor::TensorPtr InferSession::GetOutputByTensorName(const std::string &name) {}
}  // namespace mindspore
