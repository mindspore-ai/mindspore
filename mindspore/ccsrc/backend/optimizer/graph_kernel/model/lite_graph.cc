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
#include "backend/optimizer/graph_kernel/model/lite_graph.h"

#include <set>
#include <utility>

#include "backend/optimizer/graph_kernel/model/node.h"
#include "backend/optimizer/graph_kernel/model/op_node.h"
#include "backend/optimizer/graph_kernel/model/op_register.h"

namespace mindspore {
namespace opt {
namespace graphkernel {
std::string LiteGraph::Dump() const {
  std::ostringstream os;
  os << name_ << "(";
  for (size_t i = 0; i < inputs_.size(); i++) {
    os << inputs_[i]->name();
    if (i != inputs_.size() - 1) os << ", ";
  }
  os << ") -> ";
  auto &outputs = GetOutputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    os << outputs[i]->name();
    if (i != outputs.size() - 1) os << ", ";
  }
  os << " {\n";
  for (NodePtr op : ops_) {
    os << "  " << *op << "\n";
  }
  os << "}";
  return os.str();
}

const NodePtrList &LiteGraph::GetOrderedNodes() {
  std::unordered_map<NodePtr, size_t> outdegrees;
  std::function<void(NodePtr)> dfs;
  std::set<NodePtr> visited;
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
  stack.push_back(output_);
  while (!stack.empty()) {
    auto cur = stack.back();
    stack.pop_back();
    res.push_back(cur);
    for (auto &input : cur->inputs()) {
      if (input->NodeType() != NType::Primitive) continue;
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
      MS_LOG(ERROR) << "  " << *(node.first);
    }
    MS_LOG(EXCEPTION) << "Circle size: " << outdegrees.size();
  }
  std::reverse(res.begin(), res.end());
  res.pop_back();  // erase the output node
  ops_ = std::move(res);
  return ops_;
}

NodePtr LiteGraph::GraphBuilder::Emit(const std::string &op, const NodePtrList &inputs, const DAttrs &attrs,
                                      std::string node_name) {
  if (node_name.empty()) node_name = NewName();
  PrimOpPtr op_ptr = CreateOp(op, node_name);
  auto baseinfo = op_ptr->Infer(inputs, attrs);
  op_ptr->SetInputs(inputs);
  op_ptr->SetAttrs(attrs);
  op_ptr->SetBaseInfo(baseinfo);
  return graph_->Add(op_ptr);
}

NodePtr LiteGraph::GraphBuilder::Op(const std::string &op, const NodeBase &baseinfo, const NodePtrList &inputs,
                                    const DAttrs &attrs, std::string node_name) {
  if (node_name.empty()) node_name = NewName();
  PrimOpPtr op_ptr = CreateOp(op, node_name);
  op_ptr->SetInputs(inputs);
  op_ptr->SetAttrs(attrs);
  op_ptr->SetBaseInfo(baseinfo);
  return graph_->Add(op_ptr);
}

PrimOpPtr LiteGraph::GraphBuilder::CreateOp(const std::string &op, const std::string &node_name) {
  return OpRegistry::Instance().NewOp(op, node_name);
}
}  // namespace graphkernel
}  // namespace opt
}  // namespace mindspore
