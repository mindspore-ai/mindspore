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
#include "src/extendrt/tensor.h"
#include "include/model.h"
#include "ops/base_operator.h"
#include "utils/hash_map.h"
#include "include/api/status.h"

namespace mindspore {
namespace infer {
class CompileNode {
 public:
  explicit CompileNode(std::string name) : name_(std::move(name)) {}
  static CompileNode *Create(CNodePtr cnode);

  virtual ~CompileNode() = default;

  std::string GetName() const { return name_; }
  std::string GetType() const { return type_; }
  std::shared_ptr<ops::BaseOperator> GetBaseOperator() const { return base_operator_; }
  CNodePtr GetCNode() const { return cnode_; }
  const std::vector<Tensor *> &GetInputs() const { return inputs_; }
  Tensor *GetInput(size_t i) const { return inputs_.at(i); }
  size_t InputSize() const { return inputs_.size(); }
  const std::vector<Tensor *> &GetOutputs() const { return outputs_; }
  Tensor *GetOutput(size_t i) const { return outputs_.at(i); }
  size_t OutputSize() const { return outputs_.size(); }

  void SetName(const std::string &name) { name_ = name; }
  void AppendInputTensor(Tensor *tensor);
  void AppendOutputTensor(Tensor *tensor);
  void ReplaceInputTensor(Tensor *dst, Tensor *src);

  std::string Dump(int indent = 0) const;

 private:
  std::string name_{};
  std::string type_{};
  std::shared_ptr<ops::BaseOperator> base_operator_{nullptr};
  CNodePtr cnode_{nullptr};
  std::vector<Tensor *> inputs_{};
  std::vector<Tensor *> outputs_{};
};

class CompileResult {
 public:
  explicit CompileResult(Format format) : base_format_(format) {}

  virtual ~CompileResult();

  Format GetFormat() const { return base_format_; }
  CompileNode *GetNode(const std::string &name);
  CompileNode *GetArgNode(const std::string &name);
  const std::vector<CompileNode *> &GetNodes() const { return nodes_; }
  size_t NodeSize() const { return nodes_.size(); }
  const std::vector<Tensor *> &GetTensors() const { return tensors_; }
  size_t TensorSize() const { return tensors_.size(); }
  const std::vector<Tensor *> &GetInputs() const { return inputs_; }
  Tensor *GetInput(size_t i) const { return inputs_.at(i); }
  size_t InputSize() const { return inputs_.size(); }
  const std::vector<Tensor *> &GetOutputs() const { return outputs_; }
  Tensor *GetOutput(size_t i) const { return outputs_.at(i); }
  size_t OutputSize() const { return outputs_.size(); }
  const std::vector<CompileNode *> &GetParamNodes() const { return param_nodes_; }
  const std::vector<CompileNode *> &GetReturnNodes() const { return return_nodes_; }

  std::vector<CompileNode *> &GetMutableNodes();
  std::vector<Tensor *> &GetMutableInputs();
  StatusCode AppendNode(CompileNode *node);
  StatusCode AppendArgNode(CompileNode *node);
  StatusCode AppendTensor(Tensor *tensor);
  StatusCode AppendInputTensor(Tensor *tensor, bool is_borrow = false);
  StatusCode AppendOutputTensor(Tensor *tensor, bool is_borrow = false);

  StatusCode AppendNodeInputTensor(const CompileNode *compile_node, Tensor *tensor, bool is_borrow = false);
  StatusCode AppendNodeInputTensor(const std::string &node_name, Tensor *tensor, bool is_borrow = false);
  StatusCode AppendNodeOutputTensor(const CompileNode *compile_node, Tensor *tensor, bool is_borrow = false);
  StatusCode AppendNodeOutputTensor(const std::string &node_name, Tensor *tensor, bool is_borrow = false);

  void Assemble() { this->assembled_ = true; }

  std::string Dump(int indent = 0) const;

 private:
  bool assembled_ = false;
  std::vector<CompileNode *> nodes_{};
  std::vector<Tensor *> tensors_{};
  std::vector<Tensor *> inputs_{};
  std::vector<Tensor *> outputs_{};
  HashMap<std::string, CompileNode *> node_map_{};
  HashMap<std::string, Tensor *> tensor_map_{};
  std::vector<CompileNode *> param_nodes_{};
  std::vector<CompileNode *> return_nodes_{};
  std::vector<CompileNode *> arg_nodes_{};
  HashMap<std::string, CompileNode *> arg_node_map_{};
  Format base_format_{DEFAULT_FORMAT};
};
using CompileResultPtr = std::shared_ptr<CompileResult>;
}  // namespace infer
}  // namespace mindspore

#endif
