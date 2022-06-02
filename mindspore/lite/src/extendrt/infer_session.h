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
#ifndef MINDSPORE_LITE_EXTENDRT_INFER_SESSION_H
#define MINDSPORE_LITE_EXTENDRT_INFER_SESSION_H
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "ir/func_graph.h"
#include "ccsrc/backend/common/session/session_basic.h"

namespace mindspore {
class InferSession : public std::enable_shared_from_this<InferSession> {
 public:
  virtual ~InferSession() = default;
  static std::shared_ptr<InferSession> CreateSession(const std::shared_ptr<Context> context);
  virtual Status CompileGraph(FuncGraphPtr graph);
  virtual Status RunGraph();
  virtual Status Resize(const std::vector<Tensor::TensorPtr> &inputs, const std::vector<std::vector<int64_t>> &dims);

  virtual std::vector<tensor::TensorPtr> GetOutputs();
  virtual std::vector<Tensor::TensorPtr> GetInputs();
  virtual tensor::TensorPtr GetOutputByTensorName(const std::string &tensorName);
  virtual Tensor::TensorPtr GetInputByTensorName(const std::string &name);

 protected:
  InferSession() = default;
  virtual Status Init(const std::shared_ptr<Context> context);
  FuncGraphPtr graph_;
  SessionBasicPtr basic_;
  GraphId graphId_;
};
}  // namespace mindspore
#endif
