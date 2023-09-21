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

#include "pipeline/jit/graph_jit/graph_compiler/inliner/func_inliner.h"
#include <memory>
#include <string>
#include <algorithm>
#include "pipeline/jit/graph_jit/graph_compiler/parser/byte_code_parser.h"
#include "pybind11/pytypes.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace jit {
namespace graph {
namespace {
py::object GetPyFunction(const ir::NodePtr &node) {
  if (node->isa<ir::Value>()) {
    auto value = node->cast<ir::ValuePtr>()->GetValue();
    if (py::isinstance<py::function>(value)) {
      return value;
    }
  } else {
    MS_EXCEPTION_IF_CHECK_FAIL(node->isa<ir::LoadNode>(), "Expected load a function.");
    auto load = node->cast<ir::LoadNodePtr>();
    if (load->GetOpCode() == LOAD_GLOBAL) {
      return GetPyFunction(load->GetArgs().back());
    }
  }
  return py::none();
}
}  // namespace

void FuncInlineDetector::Run() { Visit(func_); }

void FuncInlineDetector::Visit_(const ir::FunctionNodePtr &node) {
  std::for_each(node->GetNodes().begin(), node->GetNodes().end(), [this](const ir::NodePtr &node) {
    cur_root_node_ = node;
    Visit(node);
    index_++;
  });
}

void FuncInlineDetector::Visit_(const ir::CallNodePtr &node) {
  auto arg = node->GetArg(0);
  if (!CanBeInlined(arg)) {
    std::for_each(node->GetArgs().begin(), node->GetArgs().end(), [this](const ir::NodePtr &node) { Visit(node); });
  } else {
    const py::object func = GetPyFunction(arg);
    auto byteCodeParser = std::make_shared<ByteCodeParser>(func);
    ir::FunctionNodePtr func_node = byteCodeParser->Parse();
    ir::NodePtrList args(node->GetArgs().begin() + 1, node->GetArgs().end());
    EvolvingFunction(func_node, args);
    node->SetArg(0, func_node);
    node_2_index_[node] = index_;
    node_2_root_[node] = cur_root_node_;
    std::for_each(node->GetArgs().begin() + 1, node->GetArgs().end(), [this](const ir::NodePtr &node) { Visit(node); });
  }
}

size_t FuncInlineDetector::GetRootNodeIndex(const ir::CallNodePtr &node) const {
  MS_EXCEPTION_IF_CHECK_FAIL(node_2_index_.find(node) != node_2_index_.end(),
                             "Invalid Call Node %" + std::to_string(node->GetNodeId()) + ".");
  return node_2_index_.at(node);
}

const ir::NodePtr &FuncInlineDetector::GetRootNode(const ir::CallNodePtr &node) const {
  MS_EXCEPTION_IF_CHECK_FAIL(node_2_root_.find(node) != node_2_root_.end(),
                             "Invalid Call Node %" + std::to_string(node->GetNodeId()) + ".");
  return node_2_root_.at(node);
}

bool FuncInlineDetector::CanBeInlined(const ir::NodePtr &node) const {
  if (!node->isa<ir::Value>() && !node->isa<ir::LoadNode>()) {
    return false;
  }
  auto func = GetPyFunction(node);
  return !py::isinstance<py::none>(func) && PyFunction_Check(func.ptr());
}

void FuncInlineDetector::EvolvingFunction(const ir::FunctionNodePtr &func_node, const ir::NodePtrList args) const {
  // Rename the local variables of the function to avoid variable name conflicts after inlining
  auto renamer = std::make_shared<FuncLocalVarRenamer>(func_node);
  renamer->Run();
  // Eliminate parameters
  auto eliminator = std::make_shared<FuncParameterEliminator>(func_node, args);
  eliminator->Run();
}

void FuncLocalVarRenamer::Run() { Visit(func_); }

void FuncLocalVarRenamer::Visit_(const ir::ParameterPtr &node) {
  node->SetName(func_->GetName() + "_" + node->GetName());
}

void FuncLocalVarRenamer::Visit_(const ir::ValuePtr &node) {
  if (node->GetScope() == ir::kScopeLocal) {
    node->SetValue(py::str(func_->GetName() + "_" + py::cast<std::string>(node->GetValue())));
  }
}

void FuncParameterEliminator::Run() { Mutate(func_); }

ir::NodePtr FuncParameterEliminator::Mutate_(const ir::ParameterPtr &node) {
  if (node->GetIndex() < args_.size()) {
    name_2_node_[node->GetName()] = args_[node->GetIndex()];
  } else {
    name_2_node_[node->GetName()] = node->GetDefaultValue();
  }
  return node;
}

ir::NodePtr FuncParameterEliminator::Mutate_(const ir::LoadNodePtr &node) {
  auto arg = node->GetArg(0);
  if (node->GetOpCode() != LOAD_FAST) {
    for (ir::NodePtr &n : node->GetArgs()) {
      n = Mutate(n);
    }
    return node;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(arg->isa<ir::Value>(), "Expected a local var name.");
  auto name = py::cast<std::string>(arg->cast<ir::ValuePtr>()->GetValue());
  if (name_2_node_.find(name) != name_2_node_.end()) {
    return name_2_node_.at(name);
  }
  return node;
}

ir::NodePtr FuncParameterEliminator::Mutate_(const ir::StoreNodePtr &node) {
  node->SetLeftArg(Mutate(node->GetLeftArg()));
  auto target = node->GetRightArg();
  if (node->GetOpCode() != STORE_FAST) {
    node->SetRightArg(Mutate(node->GetRightArg()));
    return node;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(target->isa<ir::Value>(), "Expected a local var name.");
  auto name = py::cast<std::string>(target->cast<ir::ValuePtr>()->GetValue());
  name_2_node_.erase(name);
  return node;
}

void FuncInliner::Run() {
  detector_->Run();
  Mutate(func_);
  InsertSubFunction();
}

void FuncInliner::InsertSubFunction() {
  ir::NodePtrList &roots = func_->GetNodes();
  for (auto &[index, func_node] : index_2_function_) {
    size_t idx = index + inserted_nodes_cnt_;
    roots.insert(roots.begin() + idx, func_node->GetNodes().begin(), func_node->GetNodes().end() - 1);
    inserted_nodes_cnt_ += func_node->GetNodes().size() - 1;
  }
}

ir::NodePtr FuncInliner::Mutate_(const ir::CallNodePtr &node) {
  auto func = node->GetArg(0);
  if (!func->isa<ir::FunctionNode>()) {
    return node;
  }
  auto func_node = func->cast<ir::FunctionNodePtr>();
  size_t index = detector_->GetRootNodeIndex(node) + inserted_nodes_cnt_;
  auto root = *(func_->GetNodes().begin() + index);
  MS_EXCEPTION_IF_CHECK_FAIL(root == detector_->GetRootNode(node), "Detector index error.");
  index_2_function_[index] = func_node;
  auto ret = func_node->GetNodes().back();
  MS_EXCEPTION_IF_CHECK_FAIL(ret->isa<ir::ReturnNode>(), "Excepted Return Node, but got " + ret->GetNodeName() + ".");
  return ret->cast<ir::ReturnNodePtr>()->GetReturn();
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
