/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/model/lite_graph.h"

#include <memory>
#include <algorithm>
#include <functional>
#include <set>
#include <utility>
#include <string>
#include <sstream>

#include "utils/hash_map.h"
#include "backend/common/graph_kernel/model/node.h"
#include "backend/common/graph_kernel/model/op_node.h"
#include "backend/common/graph_kernel/model/op_register.h"

namespace mindspore::graphkernel::inner {
std::string LiteGraph::ToString(bool reset_node_name) const {
  if (reset_node_name) {
    param_id_ = node_id_ = 0;
    for (auto &inp : inputs_) {
      inp->SetDebugName(ParamName());
    }
    for (auto &node : ops_) {
      node->SetDebugName(NodeName());
    }
  }
  std::ostringstream os;
  os << name_ << "(";
  for (size_t i = 0; i < inputs_.size(); i++) {
    os << inputs_[i]->debug_name();
    if (i != inputs_.size() - 1) {
      os << ", ";
    }
  }
  os << ") -> ";
  auto &outputs = GetOutputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    os << outputs[i]->debug_name();
    if (i != outputs.size() - 1) {
      os << ", ";
    }
  }
  os << " {\n";
  for (const NodePtr &op : ops_) {
    os << "  " << op->ToString() << "\n";
  }
  os << "}";
  return os.str();
}

const NodePtrList &LiteGraph::GetOrderedNodes() {
  mindspore::HashMap<NodePtr, size_t> outdegrees;
  std::function<void(NodePtr)> dfs;
  std::set<NodePtr> visited;
  // record the out degree of each nodes by Dfs.
  dfs = [&dfs, &outdegrees, &visited](const NodePtr &node) {
    (void)visited.insert(node);
    for (auto &input : node->inputs()) {
      if (input->NodeType() == NType::Primitive) {
        ++outdegrees[input];
        if (visited.count(input) == 0) {
          dfs(input);
        }
      }
    }
  };
  dfs(output_);
  NodePtrList res;
  NodePtrList stack;

  // toposort algorithm with out degree
  stack.push_back(output_);
  while (!stack.empty()) {
    auto cur = stack.back();
    stack.pop_back();
    res.push_back(cur);
    for (auto &input : cur->inputs()) {
      if (input->NodeType() != NType::Primitive) {
        continue;
      }
      --outdegrees[input];
      if (outdegrees[input] == 0) {
        stack.push_back(input);
        (void)outdegrees.erase(input);
      }
    }
  }
  if (!outdegrees.empty()) {
    MS_LOG(ERROR) << "Circle was found:";
    for (auto &node : outdegrees) {
      MS_LOG(ERROR) << "  " << node.first->debug_name();
    }
    MS_LOG(EXCEPTION) << "Circle size: " << outdegrees.size();
  }
  std::reverse(res.begin(), res.end());
  // remove the "OutputNode"
  res.pop_back();
  ops_ = std::move(res);
  return ops_;
}

PrimOpPtr CreateOp(const std::string &op, const std::string &debug_name) {
  auto node = OpRegistry::Instance().NewOp(op);
  node->SetDebugName(debug_name);
  return node;
}

NodePtr LiteGraph::GraphBuilderBase::Emit(const std::string &op, const NodePtrList &inputs, const DAttrs &attrs) const {
  PrimOpPtr op_ptr = CreateOp(op, graph_->NodeName());
  auto baseinfo = op_ptr->Infer(inputs, attrs);
  op_ptr->SetInputs(inputs);
  op_ptr->SetAttrs(attrs);
  op_ptr->SetBaseInfo(baseinfo);
  (void)graph_->ops_.emplace_back(op_ptr);
  return op_ptr;
}

NodePtr LiteGraph::GraphBuilderBase::Op(const std::string &op, const NodeBase &baseinfo, const NodePtrList &inputs,
                                        const DAttrs &attrs) const {
  PrimOpPtr op_ptr = CreateOp(op, graph_->NodeName());
  op_ptr->SetInputs(inputs);
  op_ptr->SetAttrs(attrs);
  op_ptr->SetBaseInfo({baseinfo});
  (void)graph_->ops_.emplace_back(op_ptr);
  return op_ptr;
}
}  // namespace mindspore::graphkernel::inner
