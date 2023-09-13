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

#include "src/extendrt/graph_compiler/compile_result.h"
#include <string>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>
#include "ops/base_operator.h"
#include "utils/hash_map.h"
#include "include/api/status.h"
#include "ir/primitive.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "src/common/file_utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr char tab[] = "  ";

inline std::string GenIndent(int indent) {
  std::ostringstream oss;
  for (int i = 0; i < indent; i++) {
    oss << tab;
  }
  return oss.str();
}

inline std::string DumpIntShape(const std::vector<int> &shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); i++) {
    oss << shape[i];
    if (i < shape.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

inline std::string DumpTensor(const InferTensor *tensor, int indent = 0) {
  std::ostringstream oss;
  oss << GenIndent(indent) << "Tensor <name:" << tensor->tensor_name() << ", dtype:" << tensor->data_type()
      << ", format:" << tensor->format() << ", shape:" << DumpIntShape(tensor->shape()) << ">";
  return oss.str();
}
}  // namespace

kernel::KernelAttr CompileNode::GetKernelAttr() const {
  kernel::KernelAttr attr;
  for (auto &input : inputs_) {
    (void)attr.AddInputAttr(input->data_type(), FormatEnumToString(input->format()));
  }
  for (auto &output : outputs_) {
    (void)attr.AddOutputAttr(output->data_type(), FormatEnumToString(output->format()));
  }
  return attr;
}

CompileNodePtr CompileNode::Create(CNodePtr cnode) {
  if (cnode == nullptr) {
    return nullptr;
  }
  auto primitive = GetValueNode<std::shared_ptr<Primitive>>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Node has no primitive, first input of cnode(" << cnode->fullname_with_scope()
                  << ") is : " << cnode->input(0);
    return nullptr;
  }
  auto node = std::make_shared<CompileNode>(cnode->fullname_with_scope(), kernel::PrimitiveType(primitive->name()));
  ops::PrimitiveCPtr primc{nullptr};
  if (utils::isa<ops::PrimitiveCPtr>(primitive)) {
    primc = utils::cast<ops::PrimitiveCPtr>(primitive);
  } else {
    static auto ops_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
    auto primc_creator_iter = ops_primc_fns.find(node->type_.TypeName());
    if (primc_creator_iter == ops_primc_fns.end()) {
      MS_LOG(ERROR) << "Can not find primitive_c create function for: " << node->type_;
      return nullptr;
    }
    primc = primc_creator_iter->second();
    if (primc == nullptr) {
      MS_LOG(ERROR) << "Create primitive_c failed, type: " << node->type_;
      return nullptr;
    }
    (void)primc->SetAttrs(primitive->attrs());
  }
  static auto baseops_fns = ops::OperatorRegister::GetInstance().GetOperatorMap();
  auto baseops_creator_iter = baseops_fns.find(node->type_.TypeName());
  if (baseops_creator_iter == baseops_fns.end()) {
    MS_LOG(ERROR) << "Can not find base-operator create function for: " << node->type_;
    return nullptr;
  }
  auto baseops_creator = baseops_creator_iter->second;
  node->base_operator_ = baseops_creator(primc);
  if (node->base_operator_ == nullptr) {
    MS_LOG(ERROR) << "Create base-operator failed, type: " << node->type_;
    return nullptr;
  }
  node->cnode_ = std::move(cnode);
  return node;
}

void CompileNode::AppendInputTensor(InferTensor *tensor) { this->inputs_.emplace_back(tensor); }

void CompileNode::AppendOutputTensor(InferTensor *tensor) { this->outputs_.emplace_back(tensor); }

std::string CompileNode::Dump(int indent) const {
  constexpr int kNumberTwo = 2;
  std::ostringstream oss;
  oss << GenIndent(indent) << "CompileNode <name:" << name_ << ", type:" << type_ << "> {" << std::endl;
  oss << GenIndent(indent + 1) << "inputs: [" << std::endl;
  for (auto &input : inputs_) {
    oss << DumpTensor(input, indent + kNumberTwo) << std::endl;
  }
  oss << GenIndent(indent + 1) << "]" << std::endl;
  oss << GenIndent(indent + 1) << "outputs: [" << std::endl;
  for (auto &output : outputs_) {
    oss << DumpTensor(output, indent + kNumberTwo) << std::endl;
  }
  oss << GenIndent(indent + 1) << "]" << std::endl;
  oss << GenIndent(indent) << "}";
  return oss.str();
}

void CompileNode::ReplaceInputTensor(InferTensor *dst, const InferTensor *src) {
  std::replace_if(
    inputs_.begin(), inputs_.end(), [&src](InferTensor *ele) { return ele == src; }, dst);
}

CompileNodePtr CompileResult::GetNode(const std::string &name) {
  auto iter = node_map_.find(name);
  if (iter == node_map_.end()) {
    return nullptr;
  } else {
    return iter->second;
  }
}

CompileNodePtr CompileResult::GetArgNode(const std::string &name) {
  auto iter = arg_node_map_.find(name);
  if (iter == arg_node_map_.end()) {
    return nullptr;
  } else {
    return iter->second;
  }
}

std::vector<CompileNodePtr> &CompileResult::GetMutableNodes() {
  if (assembled_) {
    MS_LOG(EXCEPTION) << "CompileResult not mutable after build.";
  }
  return nodes_;
}
std::vector<InferTensor *> &CompileResult::GetMutableInputs() {
  if (assembled_) {
    MS_LOG(EXCEPTION) << "CompileResult not mutable after build.";
  }
  return inputs_;
}

std::vector<InferTensor *> &CompileResult::GetMutableOutputs() {
  if (assembled_) {
    MS_LOG(EXCEPTION) << "CompileResult not mutable after build.";
  }
  return outputs_;
}

StatusCode CompileResult::AppendNode(CompileNodePtr node) {
  if (assembled_) {
    MS_LOG(EXCEPTION) << "CompileResult not mutable after build.";
  }
  if (node == nullptr) {
    MS_LOG(ERROR) << "Input node is nullptr";
    return kLiteInputParamInvalid;
  }
  const std::string &node_name = node->GetName();
  auto iter = node_map_.find(node_name);
  if (iter != node_map_.end()) {
    MS_LOG(ERROR) << "Duplicated node name : " << node_name;
    return kLiteError;
  }
  node_map_[node_name] = node;
  nodes_.emplace_back(node);
  return kSuccess;
}

StatusCode CompileResult::AppendArgNode(CompileNodePtr node) {
  if (assembled_) {
    MS_LOG(EXCEPTION) << "CompileResult not mutable after build.";
  }
  if (node == nullptr) {
    MS_LOG(ERROR) << "Input node is nullptr";
    return kLiteInputParamInvalid;
  }
  const std::string &node_name = node->GetName();
  auto iter = arg_node_map_.find(node_name);
  if (iter != arg_node_map_.end()) {
    MS_LOG(ERROR) << "Duplicated node name : " << node_name;
    return kLiteError;
  }
  arg_node_map_[node_name] = node;
  arg_nodes_.emplace_back(node);
  return kSuccess;
}

StatusCode CompileResult::AppendTensor(InferTensor *tensor) {
  if (assembled_) {
    MS_LOG(EXCEPTION) << "CompileResult not mutable after build.";
  }
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Input tensor is nullptr";
    return kLiteInputParamInvalid;
  }
  tensors_.emplace_back(tensor);
  return kSuccess;
}

StatusCode CompileResult::AppendInputTensor(InferTensor *tensor, bool is_borrow) {
  if (assembled_) {
    MS_LOG(EXCEPTION) << "CompileResult not mutable after build.";
  }
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Input tensor is nullptr";
    return kLiteInputParamInvalid;
  }
  inputs_.emplace_back(tensor);
  if (!is_borrow) {
    return AppendTensor(tensor);
  }
  return kSuccess;
}

StatusCode CompileResult::AppendOutputTensor(InferTensor *tensor, bool is_borrow) {
  if (assembled_) {
    MS_LOG(EXCEPTION) << "CompileResult not mutable after build.";
  }
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Input tensor is nullptr";
    return kLiteInputParamInvalid;
  }
  if (tensor->tensor_name().empty()) {
    tensor->set_tensor_name("graph_out_" + std::to_string(this->outputs_.size()));
  }
  if (!is_borrow) {
    return AppendTensor(tensor);
  }
  outputs_.emplace_back(tensor);
  return kSuccess;
}

StatusCode CompileResult::AppendNodeInputTensor(const CompileNodePtr &compile_node, InferTensor *tensor,
                                                bool is_borrow) {
  if (compile_node == nullptr) {
    MS_LOG(ERROR) << "Input compile_node is nullptr";
    return kLiteInputParamInvalid;
  }
  return AppendNodeInputTensor(compile_node->GetName(), tensor, is_borrow);
}

StatusCode CompileResult::AppendNodeInputTensor(const std::string &node_name, InferTensor *input_tensor,
                                                bool is_borrow) {
  if (assembled_) {
    MS_LOG(EXCEPTION) << "CompileResult not mutable after build.";
  }
  if (input_tensor == nullptr) {
    MS_LOG(ERROR) << "`input_tensor` is nullptr";
    return kLiteInputParamInvalid;
  }

  auto iter = node_map_.find(node_name);
  if (iter == node_map_.end()) {
    MS_LOG(ERROR) << "CompileNode not belong to this graph, node: " << node_name;
    return kLiteError;
  }
  iter->second->AppendInputTensor(input_tensor);
  if (!is_borrow) {
    return AppendTensor(input_tensor);
  }
  return kSuccess;
}

StatusCode CompileResult::AppendNodeOutputTensor(const CompileNodePtr &compile_node, InferTensor *tensor,
                                                 bool is_borrow) {
  if (compile_node == nullptr) {
    MS_LOG(ERROR) << "Input compile_node is nullptr";
    return kLiteInputParamInvalid;
  }
  return AppendNodeOutputTensor(compile_node->GetName(), tensor, is_borrow);
}

StatusCode CompileResult::AppendNodeOutputTensor(const std::string &node_name, InferTensor *output_tensor,
                                                 bool is_borrow) {
  if (assembled_) {
    MS_LOG(EXCEPTION) << "CompileResult not mutable after build.";
  }
  if (output_tensor == nullptr) {
    MS_LOG(ERROR) << "`output_tensor` is nullptr";
    return kLiteInputParamInvalid;
  }

  auto iter = node_map_.find(node_name);
  if (iter == node_map_.end()) {
    MS_LOG(ERROR) << "CompileNode not belong to this graph, node: " << node_name;
    return kLiteError;
  }
  iter->second->AppendOutputTensor(output_tensor);
  if (!is_borrow) {
    return AppendTensor(output_tensor);
  }
  return kSuccess;
}

std::string CompileResult::Dump(int indent) const {
  constexpr int kNumTwo = 2;
  std::ostringstream oss;
  oss << GenIndent(indent) << "CompileResult {" << std::endl;
  oss << GenIndent(indent + 1) << "nodes: [" << std::endl;
  for (auto &node : nodes_) {
    oss << node->Dump(indent + kNumTwo) << std::endl;
  }
  oss << GenIndent(indent + 1) << "]" << std::endl;
  oss << GenIndent(indent + 1) << "inputs: [" << std::endl;
  for (auto &input : inputs_) {
    oss << DumpTensor(input, indent + kNumTwo) << std::endl;
  }
  oss << GenIndent(indent + 1) << "]" << std::endl;
  oss << GenIndent(indent + 1) << "outputs: [" << std::endl;
  for (auto &output : outputs_) {
    oss << DumpTensor(output, indent + kNumTwo) << std::endl;
  }
  oss << GenIndent(indent + 1) << "]" << std::endl;
  oss << GenIndent(indent + 1) << "tensors: [" << std::endl;
  for (auto &tensor : tensors_) {
    oss << DumpTensor(tensor, indent + kNumTwo) << std::endl;
  }
  oss << GenIndent(indent + 1) << "]" << std::endl;
  oss << GenIndent(indent) << "}" << std::endl;
  return oss.str();
}
}  // namespace lite
}  // namespace mindspore
