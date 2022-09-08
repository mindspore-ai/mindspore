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
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/graph.h"
#include "include/api/status.h"
#include "include/common/utils/utils.h"
#include "ir/func_graph.h"
#include "backend/graph_compiler/graph_partition.h"
#include "extendrt/session/type.h"
#include "common/mutable_tensor_impl.h"
#include "extendrt/utils/kernel_graph_utils.h"

namespace mindspore {
class InferSession : public std::enable_shared_from_this<InferSession> {
 public:
  virtual ~InferSession() = default;
  static std::shared_ptr<InferSession> CreateSession(const std::shared_ptr<Context> &context);
  static SessionType SelectSession(const std::shared_ptr<Context> &context);
  virtual Status Init(const std::shared_ptr<Context> &context) = 0;
  virtual Status CompileGraph(FuncGraphPtr graph, const void *data = nullptr, size_t size = 0) = 0;
  virtual Status RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs) = 0;
  virtual Status Resize(const std::vector<tensor::Tensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
    return kSuccess;
  }

  virtual std::vector<MutableTensorImplPtr> GetOutputs() = 0;
  virtual std::vector<MutableTensorImplPtr> GetInputs() = 0;
  virtual std::vector<std::string> GetOutputNames() = 0;
  virtual std::vector<std::string> GetInputNames() = 0;
  virtual MutableTensorImplPtr GetOutputByTensorName(const std::string &tensorName) = 0;
  virtual MutableTensorImplPtr GetInputByTensorName(const std::string &name) = 0;

 protected:
  FuncGraphPtr graph_;
  compile::GraphPartitionPtr partition_;
  static void HandleContext(const std::shared_ptr<Context> &context);
};  // namespace mindspore
}  // namespace mindspore
#endif
