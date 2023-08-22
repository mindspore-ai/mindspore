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

#ifndef MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_COMPILE_RESULT_H_
#define MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_COMPILE_RESULT_H_
#include <string>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>
#include "ir/anf.h"
#include "src/infer/tensor.h"
#include "include/model.h"
#include "ops/base_operator.h"
#include "utils/hash_map.h"
#include "include/api/status.h"
#include "kernel/common_utils.h"
#include "src/infer/primitive_type.h"

namespace mindspore {
namespace lite {
class CompileNode {
 public:
  explicit CompileNode(std::string name, const kernel::PrimitiveType &type) : name_(std::move(name)), type_(type) {}
  static std::shared_ptr<CompileNode> Create(CNodePtr cnode);

  virtual ~CompileNode() = default;

  std::string GetName() const { return name_; }
  kernel::PrimitiveType GetType() const { return type_; }
  std::shared_ptr<ops::BaseOperator> GetBaseOperator() const { return base_operator_; }
  CNodePtr GetCNode() const { return cnode_; }
  const std::vector<InferTensor *> &GetInputs() const { return inputs_; }
  InferTensor *GetInput(size_t i) const { return inputs_.at(i); }
  size_t InputSize() const { return inputs_.size(); }
  const std::vector<InferTensor *> &GetOutputs() const { return outputs_; }
  InferTensor *GetOutput(size_t i) const { return outputs_.at(i); }
  size_t OutputSize() const { return outputs_.size(); }

  void SetName(const std::string &name) { name_ = name; }
  void AppendInputTensor(InferTensor *tensor);
  void AppendOutputTensor(InferTensor *tensor);
  void ReplaceInputTensor(InferTensor *dst, const InferTensor *src);
  kernel::KernelAttr GetKernelAttr() const;
  std::string Dump(int indent = 0) const;

 private:
  std::string name_{};
  kernel::PrimitiveType type_{};
  std::shared_ptr<ops::BaseOperator> base_operator_{nullptr};
  CNodePtr cnode_{nullptr};
  std::vector<InferTensor *> inputs_{};
  std::vector<InferTensor *> outputs_{};
};
using CompileNodePtr = std::shared_ptr<CompileNode>;

class CompileResult {
 public:
  CompileResult() = default;
  virtual ~CompileResult() = default;

  CompileNodePtr GetNode(const std::string &name);
  CompileNodePtr GetArgNode(const std::string &name);
  const std::vector<CompileNodePtr> &GetNodes() const { return nodes_; }
  size_t NodeSize() const { return nodes_.size(); }
  const std::vector<InferTensor *> &GetTensors() const { return tensors_; }
  size_t TensorSize() const { return tensors_.size(); }
  const std::vector<InferTensor *> &GetInputs() const { return inputs_; }
  InferTensor *GetInput(size_t i) const { return inputs_.at(i); }
  size_t InputSize() const { return inputs_.size(); }
  const std::vector<InferTensor *> &GetOutputs() const { return outputs_; }
  InferTensor *GetOutput(size_t i) const { return outputs_.at(i); }
  size_t OutputSize() const { return outputs_.size(); }
  const std::vector<CompileNodePtr> &GetParamNodes() const { return param_nodes_; }
  const std::vector<CompileNodePtr> &GetReturnNodes() const { return return_nodes_; }

  std::vector<CompileNodePtr> &GetMutableNodes();
  std::vector<InferTensor *> &GetMutableInputs();
  std::vector<InferTensor *> &GetMutableOutputs();
  StatusCode AppendNode(CompileNodePtr node);
  StatusCode AppendArgNode(CompileNodePtr node);
  StatusCode AppendTensor(InferTensor *tensor);
  StatusCode AppendInputTensor(InferTensor *tensor, bool is_borrow = false);
  StatusCode AppendOutputTensor(InferTensor *tensor, bool is_borrow = false);

  StatusCode AppendNodeInputTensor(const CompileNodePtr &compile_node, InferTensor *tensor, bool is_borrow = false);
  StatusCode AppendNodeInputTensor(const std::string &node_name, InferTensor *tensor, bool is_borrow = false);
  StatusCode AppendNodeOutputTensor(const CompileNodePtr &compile_node, InferTensor *tensor, bool is_borrow = false);
  StatusCode AppendNodeOutputTensor(const std::string &node_name, InferTensor *tensor, bool is_borrow = false);

  void Assemble() { this->assembled_ = true; }

  std::string Dump(int indent = 0) const;

 private:
  bool assembled_ = false;
  std::vector<CompileNodePtr> nodes_{};
  std::vector<InferTensor *> tensors_{};
  std::vector<InferTensor *> inputs_{};
  std::vector<InferTensor *> outputs_{};
  HashMap<std::string, CompileNodePtr> node_map_{};
  HashMap<std::string, InferTensor *> tensor_map_{};
  std::vector<CompileNodePtr> param_nodes_{};
  std::vector<CompileNodePtr> return_nodes_{};
  std::vector<CompileNodePtr> arg_nodes_{};
  HashMap<std::string, CompileNodePtr> arg_node_map_{};
};
using CompileResultPtr = std::shared_ptr<CompileResult>;
}  // namespace lite
}  // namespace mindspore

#endif
